import torch
import torch.nn as nn

# pythonのモジュールのエクスポートリスト(外から使えるようにするための物)
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

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

# 深いモデル用残差ブロック：計算効率を上げるためのブロック構造(3層)
class Bottleneck(nn.Module):

    expansion = 4 # チャネル数の増減の比率

# コンストラクタ(初期化)
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 使用する正規化層
        width = int(planes * (base_width / 64.)) * groups # 中間層(1×1と3×3)のチャネル数
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) # 畳み込み層(チャネル数の削減)
        self.bn1 = norm_layer(width) # 正規化層
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # 畳み込み層(特徴の抽出):中間層
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 畳み込み層(チャネル数の拡張)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

#順伝播の処理
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ResNetのメインクラス(ここから上のクラスにとばされることもある)
class ResNet(nn.Module):

# コンストラクタ(初期化)
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
# 初期処理層
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 7×7の畳み込み層
        self.bn1 = norm_layer(self.inplanes) # バッチ正規化
        self.relu = nn.ReLU(inplace=True) # ReLU活性化関数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 3×3の最大プーリング層
# 残差ブロック層
        self.layer1 = self._make_layer(block, 64, layers[0]) # チャネル：64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]) # チャネル：128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]) # チャネル：256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]) # チャネル：512
  # 分類層(ヘッド)
  #      self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # グローバル平均プーリング
  #      self.fc = nn.Linear(512 * block.expansion, num_classes) # 全結合層

        # 全ての層の重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual: # 恒等写像
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

# ResNetの1つの主要な残差ブロック層を構築する(残差ブロックを必要な回数積み重ねる)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
 # ダウンサンプリング処理の準備
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(x)
        c5 = self.layer4(x) # 出力をそれぞれ個別に保存して返す

#        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
#        x = self.fc(x)

        return [c2,c3,c4,c5] # 特徴マップを返す

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

# ResNet50モデルを生成するためのヘルパー関数
def resnet50(pretrained=False, progress=True, **kwargs): # (事前学習重みを使うか、それをDLのとき進捗を表示するか、その他の追加引数)
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs) # 層1～4でそれぞれ[3,4,6,3]回繰り返される