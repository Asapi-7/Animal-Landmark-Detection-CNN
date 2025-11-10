import torch
from torch import nn 
import torch.nn.functional as F 
from torch.nn import init 

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups
    )

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
            conv1x1(in_channels, out_channels)
        )

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1
    )

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.batch_norm = batch_norm

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batch_norm = batch_norm

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, depth=5, start_channels=64, batch_norm=True, up_mode='transpose', merge_mode='concat'):
        super(UNet, self).__init__()

        if up_mode not in ('transpose', 'upsample'):
            raise ValueError(f"up_mode '{up_mode}' is not a valid mode. Choose 'transpose' or 'upsample'.")
        if merge_mode not in ('concat', 'add'):
            raise ValueError(f"merge_mode '{merge_mode}' is not a valid mode. Choose 'concat' or 'add'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.start_channels = start_channels
        self.batch_norm = batch_norm
        self.up_mode = up_mode
        self.merge_mode = merge_mode

        self.down_convs = []
        self.up_convs = []

        #Encoder
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_channels * (2**i)
            pooling = i < self.depth - 1
            self.down_convs.append(DownConv(ins, outs, self.batch_norm, pooling=pooling))

        #Decoder
        for i in range(self.depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConv(ins, outs, self.batch_norm, self.merge_mode, self.up_mode))

        self.conv_final = conv1x1(self.start_channels, self.out_channels)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        #Encoder
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        #Decoder 
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x


if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    import config

    model = UNet(
        in_channels=3,
        out_channels=config.NUM_KEYPOINTS
    ).to(device)

    dummpy_input = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(device)

    output = model(dummpy_input)

    print(f"Model created successfully!")
    print(f"Input shape: {dummpy_input.shape}")
    print(f"Output shape: {output.shape}")

    expected_shape = (config.BATCH_SIZE, config.NUM_KEYPOINTS, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("Output shape is correct.")

