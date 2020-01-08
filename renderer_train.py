import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from Renderer.network import NeuralRenderer
from Renderer.stroke_gen import *
from utils.tensorboard import TensorBoard


def save_model():
    if device:
        network.cpu()
    torch.save(network.state_dict(), "renderer.pkl")
    if device:
        network.cuda()


def load_weights():
    pretrained_dict = torch.load("renderer.pkl")
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)


if __name__ == '__main__':

    writer = TensorBoard("../train_log/")
    device = torch.cuda.is_available()

    objective = nn.MSELoss()
    network = NeuralRenderer()
    adam_optim = optim.Adam(network.parameters(), lr=3e-6)
    batch_size = 64

    step_num = 0

    if os.path.isfile('renderer.pkl'):
        load_weights()
    while step_num < 500000:
        network.train()
        training_batch = []
        trues = []
        for iter in np.arange(batch_size):
            stroke_params = np.random.uniform(0, 1, 10)
            trues.append(draw(stroke_params))
            training_batch.append(stroke_params)

        training_batch = torch.tensor(training_batch).float()
        trues = torch.tensor(trues).float()
        if device:
            network = network.cuda()
            trues = trues.cuda()
            training_batch = training_batch.cuda()
        outs = network(training_batch)
        adam_optim.zero_grad()
        loss = objective(outs, trues)
        loss.backward()
        adam_optim.step()
        print(step_num, loss.item())
        if step_num < 200000:
            lr = 1e-4
        elif step_num < 400000:
            lr = 1e-5
        else:
            lr = 1e-6
        for params in adam_optim.param_groups:
            params["lr"] = lr
        writer.add_scalar("train/loss", loss.item(), step_num)
        if step_num % 100 == 0:
            network.eval()
            outs = network(training_batch)
            loss = objective(outs, trues)
            writer.add_scalar("val/loss", loss.item(), step_num)
            for iter in np.arange(32):
                out_drawing = outs[iter].cpu().data.numpy()
                true_drawing = trues[iter].cpu().data.numpy()
                writer.add_image("train/out_drawing{}.png".format(iter), out_drawing, step_num)
                writer.add_image("train/true_drawing{}.png".format(iter), true_drawing, step_num)
        if step_num % 1000 == 0:
            save_model()
        step_num += 1
