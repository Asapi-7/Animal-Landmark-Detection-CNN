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
import glob


# ----- ユーザー側ファイルをインポート -----
try:
    from resnet18_backbone import resnet18
except Exception as e:
    raise ImportError("resnet18_backbone.py から resnet18 を import できませんでした。") from e


# ----- FPN -----
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

    def forward(self, inputs):
        lateral_feats = [lateral(x) for lateral, x in zip(self.lateral_convs, inputs)]
        results = []

        x = lateral_feats[-1]
        results.append(self.output_convs[-1](x))

        for i in reversed(range(len(lateral_feats) - 1)):
            x = F.interpolate(x, scale_factor=2, mode="nearest") + lateral_feats[i]
            results.insert(0, self.output_convs[i](x))

        p6 = self.p6(results[-1])
        p7 = self.p7(F.relu(p6))
        results.extend([p6, p7])

        return {str(i): f for i, f in enumerate(results)}


# ----- Backbone wrapper -----
class BackboneWithFPN(nn.Module):
    def __init__(self, resnet_model, fpn, out_channels):
        super().__init__()
        self.body = resnet_model
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.body(x)
        if isinstance(feats, (list, tuple)) and len(feats) >= 3:
            c3, c4, c5 = feats[:3]
        else:
            raise RuntimeError("resnet18(..., return_features=True) で C3,C4,C5 が返っていません")
        return self.fpn([c3, c4, c5])


# ----- モデル構築 -----
def build_model(device, num_classes=2, pretrained_backbone=False, weights_path=None):

    resnet = resnet18(pretrained=pretrained_backbone)

    fpn_out_channels = 256
    in_channels_list = [128, 256, 512]
    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=fpn_out_channels)
    backbone = BackboneWithFPN(resnet, fpn, out_channels=fpn_out_channels)

    sizes = [8, 16, 32, 64, 128]
    anchor_generator = AnchorGenerator(
        sizes=[(s,) for s in sizes],
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = RetinaNet(backbone=backbone, num_classes=num_classes, anchor_generator=anchor_generator)
    model.to(device)

    if weights_path:
        state = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(state)
        except:
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                raise

    model.eval()
    return model


# ----- 前処理 -----
def make_transform():
    return T.Compose([T.ToTensor()])


# ----- 1枚推論 -----
def run_inference_on_image(model, image_path, device, score_thresh=0.5, draw=True, out_path=None):
    transform = make_transform()
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model([img_tensor])

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed_ms = (end - start) * 1000

    out = outputs[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    keep = scores >= score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if draw:
        draw_image_and_save(img_pil, boxes, scores, labels, out_path or "result.png")

    return boxes, scores, elapsed_ms


# ----- 描画 -----
def draw_image_and_save(pil_img, boxes, scores, labels, out_path):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), f"{scores[i]:.2f}", fill="red", font=font)

    img.save(out_path)
    print("Saved:", out_path)


# ----- 新機能：フォルダ内の複数画像で平均推論時間 -----
def measure_dir(model, dir_path, device, score_thresh):
    images = sorted(
        glob.glob(os.path.join(dir_path, "*.jpg"))
        + glob.glob(os.path.join(dir_path, "*.png"))
        + glob.glob(os.path.join(dir_path, "*.jpeg"))
    )

    if len(images) == 0:
        print("フォルダ内に画像がありません:", dir_path)
        sys.exit(1)

    print(f"{len(images)} 枚の画像で平均推論時間を測定します")

    times = []

    for img_path in images:
        transform = make_transform()
        img_tensor = transform(Image.open(img_path).convert("RGB")).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = model([img_tensor])

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        times.append(end - start)

    avg = sum(times) / len(times)
    print(f"\n平均推論時間: {avg*1000:.2f} ms")
    return avg


# ----- CLI -----
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", "-w", required=True)
    p.add_argument("--image", "-i", help="1枚だけ推論する場合")
    p.add_argument("--dir", help="複数画像で平均推論時間を測るフォルダ")
    p.add_argument("--out", "-o", help="出力画像の保存先（1枚推論モード）")
    p.add_argument("--score", "-s", type=float, default=0.5)
    p.add_argument("--device", "-d", default="cuda")
    p.add_argument("--num-classes", type=int, default=2)
    return p.parse_args()


# ----- main -----
def main():
    args = parse_args()
    device = torch.device(args.device)

    model = build_model(device, num_classes=args.num_classes,
                        pretrained_backbone=False,
                        weights_path=args.weights)

    # --- 複数画像フォルダで平均推論 ---
    if args.dir:
        measure_dir(model, args.dir, device, args.score)
        return

    # --- 1枚推論 ---
    if not args.image:
        print("画像が指定されていません (--image または --dir を指定)")
        return

    boxes, scores, elapsed_ms = run_inference_on_image(
        model, args.image, device,
        score_thresh=args.score,
        draw=True,
        out_path=args.out or "result.png"
    )

    print(f"Inference Time: {elapsed_ms:.2f} ms")
    print("Detections:", len(boxes))
    for b, s in zip(boxes, scores):
        print(f"score={s:.3f}, box={b}")


if __name__ == "__main__":
    main()
