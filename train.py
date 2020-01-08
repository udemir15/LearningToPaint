import argparse
import random
import time
import numpy as np
from DRL.evaluator import Evaluator
from DRL.ddpg import DDPG
from DRL.multi import fastenv
from utils.util import *
from utils.tensorboard import TensorBoard


def train(agent, env, evaluate):
    num_train = args.train_times
    val_int = args.validate_interval
    st_max = args.max_step
    debug = args.debug
    ep_train_count = args.episode_train_times
    out = args.output
    ts = time.time()
    st = ep = ep_st = 0
    observation = None
    noise = args.noise_factor
    while st <= num_train:
        st += 1
        ep_st += 1

        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise)
        stroke_params = agent.select_action(observation, noise_factor=noise)
        observation, reward, done, _ = env.step(stroke_params)
        agent.observe(reward, observation, done, st)
        if ep_st >= st_max and st_max:
            if st > args.warmup:
                if ep > 0 and val_int > 0 and ep % val_int == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(st - 1,
                                                                                                              np.mean(
                                                                                                                  reward),
                                                                                                              np.mean(
                                                                                                                  dist),
                                                                                                              np.var(
                                                                                                                  dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), st)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), st)
                    writer.add_scalar('validate/var_dist', np.var(dist), st)
                    agent.save_model(out)
            train_time = time.time() - ts
            ts = time.time()
            total_Q = 0.
            total_val_loss = 0.
            if st > args.warmup:
                if st < 10000 * st_max:
                    lr = (3e-4, 1e-3)
                elif st < 20000 * st_max:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in np.arange(ep_train_count):
                    Q, value_loss = agent.update_policy(lr)
                    total_Q += Q.data.cpu().numpy()
                    total_val_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], st)
                writer.add_scalar('train/actor_lr', lr[1], st)
                writer.add_scalar('train/Q', total_Q / ep_train_count, st)
                writer.add_scalar('train/critic_loss', total_val_loss / ep_train_count, st)
            if debug: prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                              .format(ep, st, train_time, time.time() - ts))
            ts = time.time()

            observation = None
            ep_st = 0
            ep += 1


if __name__ == "__main__":

    exp = os.path.abspath('.').split('/')[-1]
    writer = TensorBoard('train_log/{}'.format(exp))
    os.system('ln -sf train_log/{} ./log'.format(exp))
    if not os.path.isdir('./model'):
        os.system('mkdir ./model')

    parser = argparse.ArgumentParser(description='Learning to Paint')

    parser.add_argument('--warmup', default=400, type=int,
                        help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95 ** 5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int,
                        help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    fast_env = fastenv(args.max_step, args.env_batch, writer)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output)
    evaluator = Evaluator(args, writer)
    print('observation_space', fast_env.observation_space, 'action_space', fast_env.action_space)
    train(agent, fast_env, evaluator)
