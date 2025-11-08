import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    標準的な3層エンコーダ、3層デコーダを持つU-Netモデル
    """
    
    # --------------------------------------------------------
    # ヘルパーメソッド: 畳み込みブロックの定義
    # --------------------------------------------------------
    def conv_block(self, in_channels, out_channels):
        """
        畳み込み (3x3), バッチノーマライゼーション, ReLU を順次実行するブロック
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True はメモリ効率を改善するオプション
            
            # 通常のU-Netでは、一つのブロック内で2回の畳み込みを行うことが多い
            # 今回の疑似コードは1回に見えるが、一般的なU-Netに合わせて2回繰り返すのが自然
            # ただし、ここでは疑似コードの構造に合わせて、ブロック内に2回目の畳み込みを定義します
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    # --------------------------------------------------------
    # コンストラクタ (__init__)
    # --------------------------------------------------------
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # --- エンコーダ (ダウンサンプリング経路) ---
        # U-Netの各レベルでの特徴抽出ブロック
        self.enc1 = self.conv_block(in_channels, 64)   # 3 -> 64
        self.enc2 = self.conv_block(64, 128)           # 64 -> 128
        self.enc3 = self.conv_block(128, 256)          # 128 -> 256

        # プーリング層 (ダウンサンプリング)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- ボトルネック ---
        self.bottleneck = self.conv_block(256, 512)    # 256 -> 512

        # --- デコーダ (アップサンプリング経路) ---
        
        # 3. 最深部からのアップサンプリング
        # 転置畳み込み (ConvTranspose2d) で空間サイズを2倍にし、チャンネル数を半分にする
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        # 結合後の畳み込みブロック (スキップ接続の 256ch + upconvの 256ch = 512ch が入力)
        self.dec3 = self.conv_block(512, 256) 
        
        # 2. 中間層のアップサンプリング
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 結合後の畳み込みブロック (スキップ接続の 128ch + upconvの 128ch = 256ch が入力)
        self.dec2 = self.conv_block(256, 128)
        
        # 1. 最浅層のアップサンプリング
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 結合後の畳み込みブロック (スキップ接続の 64ch + upconvの 64ch = 128ch が入力)
        self.dec1 = self.conv_block(128, 64)

        # --- 最終出力層 ---
        # 1x1畳み込みで最終的なチャンネル数 (クラス数 or ヒートマップ数) に変換
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    # --------------------------------------------------------
    # 順伝播 (__forward__)
    # --------------------------------------------------------
    def forward(self, img):
        # 1. エンコーダパスとスキップ接続の保存
        enc1 = self.enc1(img)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # 2. ボトルネック
        bottleneck = self.bottleneck(self.pool(enc3))

        # 3. デコーダパス
        
        # レベル 3: (ボトルネックをアップコンボリューションし、enc3と結合)
        up3 = self.upconv3(bottleneck)
        # NOTE: PyTorchのConvTranspose2dは出力サイズがずれることがあるため、
        # 必要に応じてパディングやトリミング (CenterCrop/F.pad) が必要だが、
        # ここでは疑似コードのロジックに従い、単純に結合する
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        # レベル 2:
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        # レベル 1:
        up1 = self.upconv1(dec2)
        out = self.dec1(torch.cat([up1, enc1], dim=1))

        # 4. 最終出力
        return self.final_conv(out)


# --- 使用例 ---
if __name__ == '__main__':
    # 3チャンネルの入力画像を想定 (例: 224x224)
    input_tensor = torch.randn(1, 3, 224, 224) 
    
    # 9点のヒートマップを予測することを想定 (out_channels=9)
    model = UNet(in_channels=3, out_channels=9) 
    
    output = model(input_tensor)
    
    print(f"入力形状: {input_tensor.shape}")
    print(f"出力形状: {output.shape}")
    # 期待される出力形状: [1, 9, 224, 224] (バッチサイズ, チャンネル, 高さ, 幅)