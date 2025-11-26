import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url

# pythonのモジュールのエクスポートリスト(外から使えるようにするための物)
__all__ = ['ResNet', 'resnet18']

# 残差ブロック(ワープ通路を作る)を構成する畳み込み層を定義したヘルパー関数(カーネルサイズが3×3の畳み込み層)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
# ワープ通路：深い層でも情報が途切れることがないようにするためのもの

# 残差ブロック(ワープ通路を作る)を構成する畳み込み層を定義したヘルパー関数(カーネルサイズが1×1の畳み込み層)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 浅いモデル用残差ブロック：2層の畳み込みとワープ通路を持つ、残差学習の最小ユニットを定義
class BasicBlock(nn.Module):
    expansion = 1 # チャネル数の増減の比率

# コンストラクタ(初期化)
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__() # Pytorchの親クラス(nn.Module)を初期化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
# 層の定義
        self.conv1 = conv3x3(inplanes, planes, stride) # 畳み込み層
        self.bn1 = norm_layer(planes) # 正規化層
        self.relu = nn.ReLU(inplace=True) # 活性関数
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample # スキップ接続の層
        self.stride = stride

# 順伝播の処理(残差接続が実装されている)
    def forward(self, x):
        identity = x # スキップ接続を行うために入力を保持する

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) # テンソルのサイズやチャネル数が異なる場合、変換

        out += identity # 残差 F(x) + x
        out = self.relu(out)

        return out


# ResNetのメインクラス(ここから上のクラスにとばされることもある)
class ResNet(nn.Module):

# コンストラクタ(初期化)
    def __init__(self, block, layers,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # 7×7の畳み込み層
        self.bn1 = norm_layer(self.inplanes) # バッチ正規化
        self.relu = nn.ReLU(inplace=True) # ReLU活性化関数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 3×3の最大プーリング層

        self.layer1 = self._make_layer(block, 64, layers[0]) # チャネル：64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # チャネル：128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # チャネル：256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # チャネル：512

        # 全ての層の重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

      
# ResNetの1つの主要な残差ブロック層を構築する(残差ブロックを必要な回数積み重ねる)
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3) # 出力をそれぞれ個別に保存して返す

        return {'0': c1, '1': c2, '2': c3, '3': c4} # FPN用に複数の特徴マップを返す

def _resnet(block, layers, **kwargs):
    return ResNet(block, layers, **kwargs)

# resnet18
def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],**kwargs)

# FPNラッパー
from torchvision.ops import FeaturePyramidNetwork

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, fpn_out_channels=256):
        super().__init__()
        self.backbone = backbone
        in_channels_list = [64, 128, 256, 512]  # ResNet18 の各層の出力チャネル
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        features = self.backbone(x)  # 辞書
        fpn_features = self.fpn(features)
        return fpn_features
    
def resnet18_fpn(fpn_out_channels=256):
    backbone = resnet18()
    model =  BackboneWithFPN(backbone, fpn_out_channels)
    return model

