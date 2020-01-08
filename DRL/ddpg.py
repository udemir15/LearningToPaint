import numpy as np
from Renderer.network import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coordinates = torch.zeros([1, 2, 128, 128])
for i in np.arange(128):
    for j in np.arange(128):
        coordinates[0, 0, i, j] = i / 127.
        coordinates[0, 1, i, j] = j / 127.
coordinates = coordinates.to(device)

criterion = nn.MSELoss()

renderer = NeuralRenderer()
renderer.load_state_dict(torch.load('renderer.pkl'))


def decode(x, canvas):
    x = x.view(-1, 10 + 3)
    stroke_params = 1 - renderer(x[:, :10])
    stroke_params = stroke_params.view(-1, 128, 128, 1)
    color_params = stroke_params * x[:, -3:].view(-1, 1, 1, 3)
    stroke_params = stroke_params.permute(0, 3, 1, 2)
    color_params = color_params.permute(0, 3, 1, 2)
    stroke_params = stroke_params.view(-1, 5, 1, 128, 128)
    color_params = color_params.view(-1, 5, 3, 128, 128)
    for i in np.range(5):
        canvas = canvas * (1 - stroke_params[:, i]) + color_params[:, i]
    return canvas


def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)


class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40,
                 tau=0.001, discount=0.9, rmsize=800,
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size

        self.actor = ResNet(9, 18, 65)
        self.target_actor = ResNet(9, 18, 65)
        self.critic = ResNet_wobn(3 + 9, 18, 1)
        self.target_critic = ResNet_wobn(3 + 9, 18, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        if resume is not None:
            self.load_weights(resume)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.replay_mem = rpm(rmsize * max_step)

        self.tau = tau
        self.discount = discount

        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch
        self.action = [None] * self.env_batch
        self.choose_device()

    def play(self, state, is_target=False):
        state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.max_step,
                           coordinates.expand(state.shape[0], 2, 128, 128)), 1)
        if is_target:
            return self.target_actor(state)
        else:
            return self.actor(state)

    def gun_update(self, state):
        canvas = state[:, :3]
        truth = state[:, 3: 6]
        fake, real, penal = update(canvas.float() / 255, truth.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)

    def evaluate(self, state, action, target=False):
        time_step = state[:, 6: 7]
        truth = state[:, 3: 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        gan_reward = cal_reward(canvas1, truth) - cal_reward(canvas0, truth)

        coords = coordinates.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, truth, (time_step + 1).float() / self.max_step, coords], 1)

        if target:
            q_val = self.target_critic(merged_state)
            return (q_val + gan_reward), gan_reward
        else:
            q_val = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', q_val.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (q_val + gan_reward), gan_reward

    def update_policy(self, lr):
        self.log += 1

        for params in self.critic_optim.param_groups:
            params['lr'] = lr[0]
        for params in self.actor_optim.param_groups:
            params['lr'] = lr[1]

        state, action, reward, next_state, terminal = self.replay_mem.sample_batch(self.batch_size, device)

        self.gun_update(next_state)

        with torch.no_grad():
            next_stroke = self.play(next_state, True)
            target_q_network, _ = self.evaluate(next_state, next_stroke, True)
            target_q_network = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q_network

        current_q, step_reward = self.evaluate(state, action)
        target_q_network += step_reward.detach()

        value_loss = criterion(current_q, target_q_network)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.replay_mem.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        for i in np.arange(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0):
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        load_gan(path)

    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def train(self):
        self.actor.train()
        self.target_actor.train()
        self.critic.train()
        self.target_critic.train()

    def choose_device(self):
        renderer.to(device)
        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
