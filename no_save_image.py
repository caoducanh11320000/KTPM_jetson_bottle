import cv2
import threading
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# --- CẤU HÌNH ---
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"speed_log_{session_id}.txt"

MAC_IP = "192.168.2.4"
STREAM_URLS = [f"http://{MAC_IP}:5000/stream/{i}" for i in range(4)]
ENGINE_MODEL = "best.pt"
NUM_CAMS = len(STREAM_URLS)

# Khởi tạo file log nhẹ
with open(LOG_FILE, "w") as f:
    f.write(f"--- PHIÊN TỐC ĐỘ CAO: {session_id} ---\nTimestamp,Camera_ID,Status,Confidence\n")

latest_frames = [None] * NUM_CAMS
frame_locks = [threading.Lock() for _ in range(NUM_CAMS)]
last_detect_time = [0.0] * NUM_CAMS 

def receive_stream(cam_id, stream_url):
    MAX_FAILURES = 30
    while True:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            with frame_locks[cam_id]:
                latest_frames[cam_id] = None
            time.sleep(3)
            continue
        
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (480, 480))
                with frame_locks[cam_id]:
                    latest_frames[cam_id] = frame
            else:
                with frame_locks[cam_id]:
                    latest_frames[cam_id] = None
                break 
        cap.release()

def process_batch(model_path):
    # Nạp mô hình với chế độ FP16 (half=True) để giải phóng sức mạnh TensorRT
    model = YOLO(model_path, task='detect')
    time.sleep(2)
    prev_time = time.time()
    DEFECT_CLASS_ID = 0

    print("[SYSTEM] Bắt đầu chế độ siêu tốc độ (No Image Saving)...")

    while True:
        batch_frames = []
        for i in range(NUM_CAMS):
            with frame_locks[i]:
                if latest_frames[i] is not None:
                    batch_frames.append(latest_frames[i]) # Không dùng .copy() để tiết kiệm CPU
                else:
                    batch_frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
        
        # Suy luận Batch=4
        results = model(batch_frames, verbose=False, half=True)
        curr_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for cam_id, res in enumerate(results):
            for box in res.boxes:
                if int(box.cls[0]) == DEFECT_CLASS_ID:
                    detect_time = time.time()
                    if detect_time - last_detect_time[cam_id] > 2.0:
                        last_detect_time[cam_id] = detect_time
                        
                        # Chỉ ghi log text (Rất nhanh)
                        log_entry = f"{curr_timestamp},Cam_{cam_id+1},BROKEN,{float(box.conf[0]):.2f}"
                        print(f"⚠️  DETECT: {log_entry}")
                        with open(LOG_FILE, "a") as f:
                            f.write(log_entry + "\n")
                        # ĐÃ LOẠI BỎ TOÀN BỘ PHẦN VẼ BOX VÀ LƯU ẢNH .JPG

        fps_time = time.time()
        print(f"[PERFORMANCE] Total Batch FPS: {1 / (fps_time - prev_time):.1f}")
        prev_time = fps_time

if __name__ == "__main__":
    for i in range(NUM_CAMS):
        t = threading.Thread(target=receive_stream, args=(i, STREAM_URLS[i]))
        t.daemon = True
        t.start()
    process_batch(ENGINE_MODEL)
