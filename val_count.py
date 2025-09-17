import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def load_yolo_annotation(label_path, img_width, img_height):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[:5])

            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            boxes.append([x_min, y_min, x_max, y_max])

    return boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="tbank_dataset")
    parser.add_argument("--output", default="validation_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.model)
    val_images_dir = Path(args.data) / "images" / "val"
    val_labels_dir = Path(args.data) / "labels" / "val"
    output_dir = Path(args.output)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(val_images_dir.glob("*.jpg"))

    total_tp = total_fp = total_fn = 0

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        label_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_yolo_annotation(label_path, w, h)

        results = model.predict(source=image_np, conf=args.conf, verbose=False)
        pred_boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max = map(int, box[:4])
            pred_boxes.append([x_min, y_min, x_max, y_max])

        matched_gt = set()
        matched_pred = set()

        for i, gt in enumerate(gt_boxes):
            best_j = -1
            best_iou = 0.0
            for j, pred in enumerate(pred_boxes):
                if j in matched_pred:
                    continue
                iou = calculate_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= args.iou and best_j != -1:
                matched_gt.add(i)
                matched_pred.add(best_j)

        tp = len(matched_gt)
        fp = len(pred_boxes) - len(matched_pred)
        fn = len(gt_boxes) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        vis_img = image_np.copy()
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(vis_dir / f"{img_path.stem}_vis.jpg"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")

if __name__ == "__main__":
    main()