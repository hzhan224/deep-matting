import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import DatasetFromFolder
from torch.utils.data import DataLoader
from model import MattingNet

train_dset = DatasetFromFolder('data/train.txt')
train_loader = DataLoader(train_dset, 4, shuffle=True)

net = MattingNet()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for _ in range(5):
    for input, trimap, gt in train_loader:
        # tensor.size() => (batch, channel, height, width)
        inp = Variable(torch.cat((input, trimap), dim=1))
        gt = Variable(gt)

        pred = net(inp)
        loss = F.l1_loss(pred, gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.data[0])