# inference_RetinaNet18-11.py
import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms as T
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ----- ユーザー側ファイルをインポート -----
# RetinaNet18-11.py 側で使っているバックボーン実装と（もし別ファイルなら）resnet18を提供するモジュール名に合わせる
# 例: from resnet18_backbone import resnet18
# あなたの環境に合わせて下の import 行を変更してください（ファイル名と関数名が一致すること）
try:
    from resnet18_backbone import resnet18
except Exception as e:
    raise ImportError("resnet18_backbone.py から resnet18 を import できませんでした。ファイル名・関数名を確認してください。") from e

# ----- FPN クラス（学習時と同じ実装に合わせる） -----
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        self.p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):  # inputs = [C3, C4, C5]
        lateral_feats = [lateral(x) for lateral, x in zip(self.lateral_convs, inputs)]
        results = []
        x = lateral_feats[-1]
        results.append(self.output_convs[-1](x))  # P5

        for i in reversed(range(len(lateral_feats) - 1)):
            x = F.interpolate(x, scale_factor=2, mode='nearest') + lateral_feats[i]
            results.insert(0, self.output_convs[i](x))  # P4, P3

        p6 = self.p6(results[-1])  # P6
        p7 = self.p7(F.relu(p6))   # P7
        results.extend([p6, p7])

        # RetinaNet expects an OrderedDict-like mapping from feature names to tensors,
        # but the torchvision RetinaNet accepts any mapping with iterable values.
        return {str(i): f for i, f in enumerate(results)}

# ----- Backbone wrapper -----
class BackboneWithFPN(nn.Module):
    def __init__(self, resnet_model, fpn, out_channels):
        super().__init__()
        self.body = resnet_model
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        # your resnet must return (c3, c4, c5) when created with return_features=True
        feats = self.body(x)
        # If resnet returns list [c3, c4, c5] or tuple
        if isinstance(feats, (list, tuple)) and len(feats) >= 3:
            c3, c4, c5 = feats[0], feats[1], feats[2]
        else:
            # 失敗時に分かりやすくエラーを出す
            raise RuntimeError("backbone.body(x) did not return (c3,c4,c5). Make sure resnet18(..., return_features=True) is used.")
        return self.fpn([c3, c4, c5])

# ----- ヘルパー：モデル復元関数 -----
def build_model(device, num_classes=2, pretrained_backbone=False, weights_path=None):
    """
    device: torch.device
    num_classes: (背景を含まないクラス数) 例: 顔検出なら 2 (背景 + 顔) として学習しているなら num_classes=2
    pretrained_backbone: bool - resnet の pretrained フラグ（学習時と揃える）
    weights_path: 学習済み重みのパス（.pth）
    """
    # 1) バックボーン（resnet18）を return_features=True で作る
    resnet = resnet18(pretrained=pretrained_backbone)

    # 2) FPN を構築（学習時と同じチャネル数であること）
    fpn_out_channels = 256
    # ここで in_channels_list は resnet の返す feature のチャネルに合わせる必要あり
    # ResNet18 の通常のチャンネル（C3,C4,C5）は [128, 256, 512]
    in_channels_list = [128, 256, 512]
    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=fpn_out_channels)

    backbone = BackboneWithFPN(resnet, fpn, out_channels=fpn_out_channels)

    # 3) AnchorGenerator は学習時と同じ設定にする
    # 学習時に使用した sizes と aspect_ratios を合わせること
    # ここはあなたの学習スクリプトに合わせて調整済み（例: sizes=[8,16,32,...] を feature map 数だけ使う）
    sizes = [8, 16, 32, 64, 128, 224]
    # num_feature_maps は FPN の出力数（P3,P4,P5,P6,P7） => 5
    num_feature_maps = 5
    sizes_for_anchor = tuple((s,) for s in sizes[:num_feature_maps])
    anchor_generator = AnchorGenerator(sizes=sizes_for_anchor, aspect_ratios=((0.5, 1.0, 2.0),) * num_feature_maps)

    # 4) RetinaNet を作る
    model = RetinaNet(backbone=backbone, num_classes=num_classes, anchor_generator=anchor_generator)
    model.to(device)

    # 5) 重みロード（存在すれば）
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        # もし学習時に model.state_dict() を直接保存しているなら load_state_dict を使う
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # 場合によっては state が {'model_state_dict': ...} のようになっているかもしれない
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                # 失敗したらそのまま例外を投げる
                raise
    model.eval()
    return model

# ----- 画像前処理（学習時と揃える：ここは学習時に使った処理に合わせること） -----
def make_transform():
    return T.Compose([T.ToTensor()])

# ----- 推論処理 -----
def run_inference_on_image(model, image_path, device, score_thresh=0.5, draw=True, out_path=None):
    transform = make_transform()
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).to(device)
    start_time = time.time()

    # モデルは list[Tensor] を受け取ることに注意
    with torch.no_grad():
        outputs = model([img_tensor])

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000.0  # ミリ秒

    if len(outputs) == 0:
        return [], [], inference_time

    out = outputs[0]
    boxes = out.get("boxes", torch.zeros((0,4))).cpu().numpy()
    scores = out.get("scores", torch.zeros((0,))).cpu().numpy()
    labels = out.get("labels", torch.zeros((0,), dtype=torch.int64)).cpu().numpy()

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if draw:
        draw_image_and_save(img_pil, boxes, scores, labels, out_path or f"{os.path.splitext(image_path)[0]}_pred.png")

    return boxes, scores, inference_time

# ----- 描画関数 -----
def draw_image_and_save(pil_img, boxes, scores, labels, out_path):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{scores[i]:.2f}"
        if font:
            draw.text((x1+2, y1+2), text, fill="red", font=font)
        else:
            draw.text((x1+2, y1+2), text, fill="red")
    img.save(out_path)
    print(f"Saved visualization to: {out_path}")

# ----- CLI -----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", "-w", required=True, help="Path to weights file (e.g., retinanet18112_weights_SGD.pth)")
    p.add_argument("--image", "-i", required=True, help="Input image file (e.g., hyrax_4.jpg)")
    p.add_argument("--out", "-o", default=None, help="Output image file path (e.g., hyrax_result.png)")
    p.add_argument("--score", "-s", type=float, default=0.5, help="Score threshold (default: 0.5)")
    p.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--num-classes", type=int, default=2, help="Number of classes (default: 2)")
    return p.parse_args()

def measure_average_inference_time(model, image_path, device, score_thresh=0.5, repeat=50):
        transform = make_transform()
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = transform(img_pil).to(device)

    # ----- Warm-up -----
        for _ in range(5):
            _ = model([img_tensor])
            if device.type == "cuda":
                torch.cuda.synchronize()

    # ----- Repeat推論 -----
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = model([img_tensor])
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000.0)  # ms

        avg_ms = sum(times) / len(times)
        return avg_ms

def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(device, num_classes=args.num_classes, pretrained_backbone=False, weights_path=args.weights)

    boxes, scores, inference_time = run_inference_on_image(
        model, args.image, device, score_thresh=args.score, draw=True, out_path=args.out
    )

    avg_time = measure_average_inference_time(model, args.image, device)
    print(f"Average inference time (50 runs): {avg_time:.2f} ms")

    print(f"Inference Time: {inference_time:.2f} ms")
    print("Detections:", len(boxes))
    for b, s in zip(boxes, scores):
        print(f"score={s:.3f}, box={b}")


if __name__ == "__main__":
    main()
