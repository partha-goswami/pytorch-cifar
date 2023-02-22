import torch
import torch.nn as nn


class UltimusBlock(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        Init Method

        :param in_features: Input Features
        :param out_features: Output Features
        '''
        super(UltimusBlock, self).__init__()
        self.fc_k = nn.Linear(in_features, out_features * 8)
        self.fc_q = nn.Linear(in_features, out_features * 8)
        self.fc_v = nn.Linear(in_features, out_features * 8)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Linear(out_features * 8, out_features)

    def forward(self, x):
        '''
        Forward Method

        :param x: Input
        :return: Linear layer Output
        '''
        k = self.fc_k(x)
        q = self.fc_q(x)
        v = self.fc_v(x)

        k = k.view(k.size(0), 8, -1)
        q = q.view(q.size(0), 8, -1)
        v = v.view(v.size(0), 8, -1)

        am = self.softmax(torch.bmm(q.transpose(1, 2), k) / 8 ** 0.5)
        z = torch.bmm(am, v).view(v.size(0), -1)
        out = self.out(z)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        '''
        Input: 32X32X3,
        Filters: 16
        Output: (32+(2*1)-3)/1 + 1 => 32X32X16
        '''
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        '''
        Input: 32X32X16,
        Filters: 32
        Output: (32+(2*1)-3)/1 + 1 => 32X32X32
        '''
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        '''
        Input: 32X32X32,
        Filters: 48
        Output: (32+(2*1)-3)/1 + 1 => 32X32X48
        '''
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)

        '''
        Input: 32X32X48,
        Output: 1X1X48
        '''
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        '''
        Ultimus Blocks
        '''
        self.ultimus_block1 = UltimusBlock(48, 48)
        self.ultimus_block2 = UltimusBlock(48, 48)
        self.ultimus_block3 = UltimusBlock(48, 48)
        self.ultimus_block4 = UltimusBlock(48, 48)
        self.fc_out = nn.Linear(48, 10)

    def forward(self, x):
        '''
        Forward Method of Main Network that utilizes the ultimus block

        :param x: Input
        :return: Linear layer output of 10 classes
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 48)
        x = self.ultimus_block1(x)
        x = self.ultimus_block2(x)
        x = self.ultimus_block3(x)
        x = self.ultimus_block4(x)
        x = self.fc_out(x)
        return x
