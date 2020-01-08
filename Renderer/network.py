import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralRenderer(nn.Module):

    def __init__(self):
        super(NeuralRenderer, self).__init__()
        self.lin1 = nn.Linear(10, 512)
        self.lin2 = nn.Linear(512, 1024)
        self.lin3 = nn.Linear(1024, 2048)
        self.lin4 = nn.Linear(2048, 4096)
        self.cnn1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.cnn4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.cnn6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pix_shuf = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.cnn1(x))
        x = self.pix_shuf(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = self.pix_shuf(self.cnn4(x))
        x = F.relu(self.cnn5(x))
        x = self.pix_shuf(self.cnn6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)
