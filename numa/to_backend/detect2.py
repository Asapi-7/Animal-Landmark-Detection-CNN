import os
import torch
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F

#bbx
from torchvision.models import resnet18
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou
import torch.optim as optim

#lmks
from resnet_map4 import LandmarkHeatmapRegressor
from resnet_map4 import IMG_SIZE, NUM_LANDMARKS as NUM_KEYPOINTS


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

#bbx
def build_faster_rcnn_model(num_classes: int, weight_path: str, device: str):
    try:
        backbone = resnet_fpn_backbone('resnet18', pretrained=True)
        backbone.out_channels = 256

        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        NUM_CLASSES = 2
        model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator)

        print(f"✅ Loading Faster R-CNN weights: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Faster R-CNNロードエラー: {e}")
        return None

#lmks

def build_landmark_resnet_model(num_keypoints: int, weight_path: str, device: str):
    try:
        model = LandmarkHeatmapRegressor(num_landmarks=num_keypoints, pretrained=False).to(device)
        print(f"Loading ResNet heatmap weights: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"ResNet heatmap model load failed: {e}")
        return None


def load_ml_model():
    # Faster R-CNN
    FASTER_RCNN_WEIGHTS = "fasterrcnn4_resnet18_W2_epoch_20.pth" 
    global face_detector, landmark_detector, RESNET_TRANSFORM
    face_detector = build_faster_rcnn_model(num_classes=2, weight_path=FASTER_RCNN_WEIGHTS, device=DEVICE)

    # lmks
    LANDMARK_WEIGHTS = "model_map4.pth" 
    landmark_detector = build_landmark_resnet_model(NUM_KEYPOINTS, LANDMARK_WEIGHTS, DEVICE)

    RESNET_TRANSFORM = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

def decode_heatmaps_to_coordinates(heatmaps: torch.Tensor) -> np.ndarray:
    _, num_keypoints, H, W = heatmaps.shape
    
    flat_heatmaps = heatmaps.view(1, num_keypoints, -1)
    max_indices = torch.argmax(flat_heatmaps, dim=2)
    y_coords = max_indices // W
    x_coords = max_indices % W
    
    x_coords = x_coords * (IMG_SIZE / W)
    y_coords = y_coords * (IMG_SIZE / H) 
    
    coordinates = torch.stack((x_coords, y_coords), dim=2).squeeze(0) 
    return coordinates.numpy()

def detect_face_and_lndmk(image_path: str, score_threshold: float = 0.3):
    if face_detector is None or landmark_detector is None:
        print("❌ 必要なモデルがロードされていません。")
        return None

    try:
        img_original = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return None
    
    W, H = img_original.size
    
    
    img_tensor = F.to_tensor(img_original).to(DEVICE)

    with torch.no_grad():
        outputs = face_detector([img_tensor])

    output = outputs[0]
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    
    if len(scores) == 0:
        print("⚠️ No detections found.")
        return None
    
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    best_score = scores[best_idx]

    if best_score < score_threshold:
        print(f"⚠️ No box above threshold ({best_score:.2f} < {score_threshold})")
        return None
    
    xmin, ymin, xmax, ymax = best_box.astype(float)
    
    xmin_int = max(0, int(xmin))
    ymin_int = max(0, int(ymin))
    xmax_int = min(W, int(xmax))
    ymax_int = min(H, int(ymax))

    
    cropped_img_pil = img_original.crop((xmin_int, ymin_int, xmax_int, ymax_int))
    crop_W, crop_H = cropped_img_pil.size
    
    if crop_W <= 0 or crop_H <= 0:
        print("⚠️ クロップされた領域のサイズが不正です。")
        return None
    #lmks    
    cropped_img_tensor = RESNET_TRANSFORM(cropped_img_pil).unsqueeze(0).to(DEVICE)
   
    with torch.no_grad():
        heatmaps_pred = landmark_detector(cropped_img_tensor).cpu()
        
    relative_landmarks_np = decode_heatmaps_to_coordinates(heatmaps_pred)

    scale_x = crop_W / IMG_SIZE 
    scale_y = crop_H / IMG_SIZE 
    
    absolute_landmarks_np = relative_landmarks_np.copy()
    absolute_landmarks_np[:, 0] *= scale_x
    absolute_landmarks_np[:, 1] *= scale_y

    absolute_landmarks_np[:, 0] += xmin_int
    absolute_landmarks_np[:, 1] += ymin_int

    
    output_list = [
        [float(xmin), float(ymin)], 
        [float(xmax), float(ymax)]
    ]
    
    for x, y in absolute_landmarks_np:
        output_list.append([float(x), float(y)])
        
    print(f"✅ 検出完了。BBoxスコア: {best_score:.2f}")
    return output_list


if __name__ == "__main__":
    test_image = "test.jpg" # 適切な画像パスに変更
    SCORE_THRESHOLD = 0.3

    print("\n--- モデルロード中 ---")
    load_ml_model()
    print("\n--- 推論開始 ---")
    result = detect_face_and_lndmk(test_image, score_threshold=SCORE_THRESHOLD)
    
    if result is not None:
        print(f"要素数: {len(result)}")
        print(f"BBox (左上): {result[0]}")
        print(f"BBox (右下): {result[1]}")
        print(f"ランドマーク (9点) : {result[2:]}") #ここの9点がランドマークの座標
    else:
        print("❌ 処理失敗、または閾値未満の検出結果でした。")


#from detect import load_ml_model, detect_face_and_lndmk