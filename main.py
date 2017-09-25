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
    for input, trimap, gt, fg, bg in train_loader:
        # tensor.size() => (batch, channel, height, width)
        inp = Variable(torch.cat((input, trimap), dim=1))  #baozhuangcheng bianliang is the rule of pytorch
        gt = Variable(gt)

        pred = net(inp)                  #input inp to the network and get the predict alph value
        loss1 = F.l1_loss(pred, gt)       #l1-MAE l2-MSE l1:|predict alph value -gt alph value |

        res = pred * fg + (1 - pred) * bg
        loss2 = F.mse_loss(res, input)

        loss = 0.5 * loss1 + 0.5 * loss2

        opt.zero_grad()                  #all the grad get value:0
        loss.backward()                  #fanxiangqiudao
        opt.step()                       #go one step

        print(loss.data[0])
