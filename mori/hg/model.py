
import torch
from torch import nn

Pool = nn.MaxPool2d

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "input channel {} doesn't match layer's input {} ".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2

class HourglassNet(nn.Module):
    """ 
    Stacked Hourglass Network for landmark detection.
    """
    def __init__(self, nstack, nclasses, nfeats, bn=False, increase=0):
        super(HourglassNet, self).__init__()
        self.nstack = nstack
        self.nclasses = nclasses
        self.nfeats = nfeats

        # Input processing
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeats)
        )

        # Hourglass modules
        self.hgs = nn.ModuleList([Hourglass(4, nfeats, bn, increase) for _ in range(nstack)])

        # Feature extraction layers that follow each hourglass module
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(nfeats, nfeats),
                Conv(nfeats, nfeats, 1, bn=True, relu=True)
            ) for _ in range(nstack)
        ])

        # Output layers to produce heatmaps
        self.outs = nn.ModuleList([Conv(nfeats, nclasses, 1, relu=False, bn=False) for _ in range(nstack)])
        
        # Layers to merge features and predictions for the next stack
        self.merge_features = nn.ModuleList([Conv(nfeats, nfeats, 1, relu=False, bn=False) for _ in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Conv(nclasses, nfeats, 1, relu=False, bn=False) for _ in range(nstack - 1)])

    def forward(self, x):
        x = self.pre(x)
        
        outputs = []
        for i in range(self.nstack):
            hg_out = self.hgs[i](x)
            feature_out = self.features[i](hg_out)
            preds = self.outs[i](feature_out)
            outputs.append(preds)

            # If not the last stack, prepare input for the next stack
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature_out)
        
        return outputs
