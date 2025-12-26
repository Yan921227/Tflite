# -*- coding: utf-8 -*-
"""
完整揮拍分析 Pipeline (TFLite 版本)

功能：
1. 用 YOLO (TFLite) 從影片中找出 backswing(拉拍) / impact(擊球) 幀
2. 分群 & 配對，計算每一次揮拍
3. 擷取那些幀為圖片
4. 對圖片跑 MediaPipe Pose，計算關節角度
5. 輸出 result.json

環境需求：
- ultralytics
- tensorflow (或 ai-edge-litert)
- opencv-python
- mediapipe
- pillow
- numpy
"""

import math
import json
import csv
from pathlib import Path
import os

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

# ==============================
# 0. 設定區（請依實際環境修改）
# ==============================
VIDEO_PATH = "C:\\Users\\User\\Desktop\\IMG_1726_sdr.mp4"   # 影片路徑

# ★★★ 修改重點 1：這裡改成你的 .tflite 檔案路徑 ★★★
# 請確認這個檔案真的存在於這個路徑下
MODEL_PATH = "train7\\weights\\best_saved_model\\best_float32.tflite"

OUTPUT_DIR = Path("pipeline_output\\IMG_1726_sdr_tflite")        # 輸出改個名字區分一下

# YOLO 信心度設定
BASE_CONFIDENCE = 0.25
CLASS_CONFIDENCE = {
    0: 0.50,  # backswing
    1: 0.50,  # impact
}

# 以秒為單位
BACKSWING_GROUP_WINDOW_SEC = 0.10   
IMPACT_GROUP_WINDOW_SEC   = 0.10   
MAX_SWING_DURATION_SEC    = 1.00   


# MediaPipe 33 點名稱
MP_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

# 要計算的關節
JOINTS = {
    "Right Elbow":    ("RIGHT_SHOULDER",  "RIGHT_ELBOW",   "RIGHT_WRIST"),
    "Left Elbow":     ("LEFT_SHOULDER",   "LEFT_ELBOW",    "LEFT_WRIST"),
    "Right Shoulder": ("RIGHT_ELBOW",     "RIGHT_SHOULDER","RIGHT_HIP"),
    "Left Knee":      ("LEFT_HIP",        "LEFT_KNEE",     "LEFT_ANKLE"),
    "Right Knee":     ("RIGHT_HIP",       "RIGHT_KNEE",    "RIGHT_ANKLE"),
}

LABELS_FOR_JOINT = {
    "Right Elbow": "A",
    "Left Elbow": "B",
    "Right Shoulder": "C",
    "Left Knee": "D",
    "Right Knee": "E",
}

TEXT_OFFSET = {
    "Right Elbow":    (15, -30),
    "Left Elbow":     (15, -30),
    "Right Shoulder": (10, 20),
    "Left Knee":      (-40, -20),
    "Right Knee":     (-40, -20),
}


# ==============================
# 1. YOLO 偵測揮拍幀 (TFLite 版)
# ==============================

def group_frames(candidates, max_gap):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["frame"])
    groups = []
    current = [candidates[0]]

    for c in candidates[1:]:
        if c["frame"] - current[-1]["frame"] <= max_gap:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
    groups.append(current)

    rep_frames = []
    for g in groups:
        best = max(g, key=lambda x: x["conf"])
        rep_frames.append(best["frame"])
    return rep_frames


def pair_swings(backswing_frames, impact_frames, max_gap):
    backswing_frames = sorted(backswing_frames)
    impact_frames = sorted(impact_frames)

    swings = []
    i = 0 

    for bs in backswing_frames:
        while i < len(impact_frames) and impact_frames[i] < bs:
            i += 1
        if i >= len(impact_frames):
            break
        ip = impact_frames[i]
        if ip - bs <= max_gap:
            swings.append({"backswing": bs, "impact": ip})
            i += 1

    return swings


def detect_swings(video_path, model_path):
    # 檢查 TFLite 檔案是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型檔案：{model_path}\n請檢查路徑是否正確！")

    print(f"🔄 載入 TFLite 模型: {model_path}")
    # Ultralytics 會自動識別 .tflite 格式並呼叫 TensorFlow
    model = YOLO(model_path, task='detect') 

    print(f"🔄 開啟影片: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError("無法開啟影片，請確認路徑。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"📊 影片資訊: 幀數={total_frames}, FPS={fps:.2f}")

    frame_idx = 0
    backswing_candidates = []
    impact_candidates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ★★★ 修改重點 2：加上 imgsz=640 ★★★
        # TFLite 模型輸入大小是固定的，指定 imgsz 可以確保轉換正確
        results = model(frame, imgsz=640, conf=BASE_CONFIDENCE, verbose=False)
        
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            threshold = CLASS_CONFIDENCE.get(cls_id, BASE_CONFIDENCE)
            if conf < threshold:
                continue

            if cls_id == 0:  # backswing
                backswing_candidates.append({"frame": frame_idx, "conf": conf})
            elif cls_id == 1:  # impact
                impact_candidates.append({"frame": frame_idx, "conf": conf})

    cap.release()

    print(f"候選 backswing 幀數量: {len(backswing_candidates)}")
    print(f"候選 impact 幀數量: {len(impact_candidates)}")

    backswing_frames = group_frames(backswing_candidates, int(BACKSWING_GROUP_WINDOW_SEC * fps))
    impact_frames = group_frames(impact_candidates, int(IMPACT_GROUP_WINDOW_SEC * fps))

    print("代表 backswing 幀:", backswing_frames)
    print("代表 impact 幀:", impact_frames)

    swings = pair_swings(backswing_frames, impact_frames, int(MAX_SWING_DURATION_SEC * fps))
    print(f"👉 偵測到 {len(swings)} 次揮拍")
    
    return swings, fps


# ==============================
# 2. 從影片擷取特定幀成圖片
# ==============================

def save_frames_from_video(video_path, frame_indices, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError("無法開啟影片，請確認路徑。")

    saved_paths = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"[警告] 影片中讀不到 frame {idx}")
            continue

        out_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_paths[idx] = str(out_path)
        print(f"已儲存 frame {idx} -> {out_path}")

    cap.release()
    return saved_paths


# ==============================
# 3. MediaPipe + 關節角度計算
# ==============================

def calculate_angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    mag_BA = math.hypot(*BA)
    mag_BC = math.hypot(*BC)
    if mag_BA == 0 or mag_BC == 0:
        return None
    cos_t = max(min(dot / (mag_BA * mag_BC), 1.0), -1.0)
    return math.degrees(math.acos(cos_t))


def extract_landmarks(result, width, height):
    if not result.pose_landmarks:
        return {}
    lm = result.pose_landmarks.landmark
    landmarks = {}
    for idx, p in enumerate(lm):
        if idx >= len(MP_NAMES):
            continue
        name = MP_NAMES[idx]
        x_px = int(round(p.x * width))
        y_px = int(round(p.y * height))
        landmarks[name] = (x_px, y_px)
    return landmarks


def compute_joint_angles(landmarks):
    angles = {}
    for joint_name, (pA, pB, pC) in JOINTS.items():
        if pA in landmarks and pB in landmarks and pC in landmarks:
            A, B, C = landmarks[pA], landmarks[pB], landmarks[pC]
            ang = calculate_angle(A, B, C)
            angles[joint_name] = ang
        else:
            angles[joint_name] = None
    return angles


def draw_angles_on_image(image_bgr, landmarks, angles):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.load_default()
    # 這裡你可以嘗試載入更好的字型，如果有的話

    r = 8
    for joint_name, (pA, pB, pC) in JOINTS.items():
        if pA in landmarks and pB in landmarks and pC in landmarks:
            A, B, C = landmarks[pA], landmarks[pB], landmarks[pC]
            for P in (A, B, C):
                draw.ellipse([P[0]-r, P[1]-r, P[0]+r, P[1]+r], fill=(255, 0, 0))
            dx, dy = TEXT_OFFSET.get(joint_name, (0, -35))
            label = LABELS_FOR_JOINT.get(joint_name, "?")
            draw.text((B[0] + dx, B[1] + dy), label, font=font, fill=(255, 0, 0))

    annotated_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return annotated_bgr


def process_image_with_mediapipe(image_path, output_image_path=None, output_csv_path=None):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"無法讀取圖片：{image_path}")
    h, w = image_bgr.shape[:2]

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

    landmarks = extract_landmarks(result, w, h)
    angles = compute_joint_angles(landmarks)

    print(f"圖片 {image_path} 各關節角度：")
    for joint_name in JOINTS.keys():
        ang = angles[joint_name]
        print(f"  {joint_name}: {ang if ang else 'N/A'}")

    annotated = draw_angles_on_image(image_bgr, landmarks, angles)
    if output_image_path:
        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

    if output_csv_path:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Joint", "Label", "Angle (degrees)"])
            for joint_name in JOINTS.keys():
                label = LABELS_FOR_JOINT.get(joint_name, "?")
                ang = angles[joint_name]
                writer.writerow([joint_name, label, f"{ang:.1f}" if ang else "N/A"])

    return angles


# ==============================
# 4. 串成完整 Pipeline
# ==============================

def full_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # (1) 用 YOLO (TFLite) 偵測
    swings, fps = detect_swings(VIDEO_PATH, MODEL_PATH)

    if not swings:
        print("⚠ 沒有偵測到揮拍。")
        return

    # (2) 擷取圖片
    all_frames = set()
    for s in swings:
        all_frames.add(s["backswing"])
        all_frames.add(s["impact"])

    frames_dir = OUTPUT_DIR / "frames"
    frame_to_path = save_frames_from_video(VIDEO_PATH, all_frames, frames_dir)

    # (3) MediaPipe + 輸出
    visual_dir = OUTPUT_DIR / "visual"
    csv_dir = OUTPUT_DIR / "csv"

    result_for_app = {
        "video": Path(VIDEO_PATH).name,
        "fps": fps,
        "swings": []
    }

    for i, s in enumerate(swings, 1):
        bs_frame = s["backswing"]
        ip_frame = s["impact"]

        if bs_frame not in frame_to_path or ip_frame not in frame_to_path:
            continue

        bs_img_path = frame_to_path[bs_frame]
        ip_img_path = frame_to_path[ip_frame]

        bs_out_img = visual_dir / f"swing_{i:02d}_backswing.jpg"
        ip_out_img = visual_dir / f"swing_{i:02d}_impact.jpg"
        bs_out_csv = csv_dir / f"swing_{i:02d}_backswing.csv"
        ip_out_csv = csv_dir / f"swing_{i:02d}_impact.csv"

        print(f"\n=== 處理 Swing {i} backswing (frame {bs_frame}) ===")
        bs_angles = process_image_with_mediapipe(bs_img_path, bs_out_img, bs_out_csv)

        print(f"\n=== 處理 Swing {i} impact (frame {ip_frame}) ===")
        ip_angles = process_image_with_mediapipe(ip_img_path, ip_out_img, ip_out_csv)

        result_for_app["swings"].append({
            "index": i,
            "backswing": {"frame": int(bs_frame), "angles": bs_angles},
            "impact": {"frame": int(ip_frame), "angles": ip_angles},
        })

    json_path = OUTPUT_DIR / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_for_app, f, ensure_ascii=False, indent=2)

    print("\n✅ 全部完成 (使用 TFLite 模型)")
    print(f"📁 輸出：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    full_pipeline()