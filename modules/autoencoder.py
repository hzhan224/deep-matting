import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def block(self, nb, inC, outC, ks=5):
        layers = [nn.Conv2d(inC, outC, ks, 1, ks//2), 
                  nn.ReLU()]
        for _ in range(nb-1):
            layers.append(nn.Conv2d(outC, outC, ks, 1, ks//2))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def __init__(self):
        super().__init__()
        self.e1 = self.block(2, 4, 64)
        self.e2 = self.block(2, 64, 128)
        self.e3 = self.block(3, 128, 256)
        self.e4 = self.block(3, 256, 512)
        self.e5 = self.block(3, 512, 512)
        self.e6 = self.block(1, 512, 512, ks=1)

        self.d6 = self.block(1, 512, 512, ks=1)
        self.d5 = self.block(1, 512, 512)
        self.d4 = self.block(1, 512, 256)
        self.d3 = self.block(1, 256, 128)
        self.d2 = self.block(1, 128, 64)
        self.d1 = self.block(1, 64, 64)
        self.bottle = nn.Conv2d(64, 1, 5, 1, 2)
    
    def forward(self, input):
        x = self.e1(input)
        x, ind1 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.e2(x)
        x, ind2 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.e3(x)
        x, ind3 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.e4(x)
        x, ind4 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.e5(x)
        x, ind5 = F.max_pool2d(x, 2, 2, return_indices=True)
        x = self.e6(x)
        x = self.d6(x)
        x = F.max_unpool2d(x, ind5, 2, stride=2)
        x = self.d5(x)
        x = F.max_unpool2d(x, ind4, 2, stride=2)
        x = self.d4(x)
        x = F.max_unpool2d(x, ind3, 2, stride=2)
        x = self.d3(x)
        x = F.max_unpool2d(x, ind2, 2, stride=2)
        x = self.d2(x)
        x = F.max_unpool2d(x, ind1, 2, stride=2)
        x = self.d1(x)
        x = self.bottle(x)

        return x


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch.optim import Adam
    # (batch, channel, height, width)
    crit = nn.MSELoss()
    net = AutoEncoder()
    opt = Adam(net.parameters(), lr=1e-3)

    for batch in range(5):
        input = torch.rand(2, 4, 64, 64)
        input = Variable(input)
        label = torch.rand(2, 1, 64, 64)
        label = Variable(label)

        output = net(input)
        loss = crit(output, label)
        print(loss.data[0])

        opt.zero_grad()
        loss.backward()
        opt.step()




