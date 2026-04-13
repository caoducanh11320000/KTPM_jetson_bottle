import cv2
import threading
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# --- KHỞI TẠO SESSION ID (Duy nhất cho mỗi lần chạy) ---
# Định dạng: NămThángNgày_GiờPhútGiây (Ví dụ: 20260413_083015)
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- CẤU HÌNH ĐƯỜNG DẪN DYNAMIC ---
LOG_FILE = f"defect_log_{session_id}.txt"
SAVE_DIR = f"defect_images_{session_id}"

# Cấu hình kết nối
MAC_IP = "192.168.2.4" 
STREAM_URLS = [f"http://{MAC_IP}:5000/stream/{i}" for i in range(4)]
ENGINE_MODEL = "best.pt"
NUM_CAMS = len(STREAM_URLS)

# Tự động tạo thư mục và file log mới cho mỗi phiên
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"[SYSTEM] Đã tạo thư mục lưu ảnh: {SAVE_DIR}")

with open(LOG_FILE, "w") as f:
    f.write(f"--- PHIÊN LÀM VIỆC: {session_id} ---\n")
    f.write("Timestamp,Camera_ID,Status,Confidence\n")
    print(f"[SYSTEM] Đã tạo file log mới: {LOG_FILE}")

# --- BIẾN ĐIỀU KHIỂN ĐA LUỒNG ---
latest_frames = [None] * NUM_CAMS
frame_locks = [threading.Lock() for _ in range(NUM_CAMS)]

def receive_stream(cam_id, stream_url):
    cap = cv2.VideoCapture(stream_url)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (480, 480))
            with frame_locks[cam_id]:
                latest_frames[cam_id] = frame
        else:
            time.sleep(0.1)

def process_batch(model_path):
    # Nạp mô hình với cấu hình FP16 để tối ưu GPU Orin
    model = YOLO(model_path, task='detect')
    time.sleep(2)
    prev_time = time.time()
    
    # ID của lớp lỗi (Broken) - Bạn hãy check lại metadata nếu cần
    DEFECT_CLASS_ID = 0 

    while True:
        batch_frames = []
        for i in range(NUM_CAMS):
            with frame_locks[i]:
                if latest_frames[i] is not None:
                    batch_frames.append(latest_frames[i].copy())
                else:
                    batch_frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
        
        # Suy luận đồng thời 4 camera (Batch=4)
        results = model(batch_frames, verbose=False, half=True)
        
        curr_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for cam_id, res in enumerate(results):
            boxes = res.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == DEFECT_CLASS_ID:
                    log_entry = f"{curr_timestamp},Cam_{cam_id+1},BROKEN,{conf:.2f}"
                    print(f"⚠️  PHÁT HIỆN LỖI: {log_entry}")
                    
                    # Ghi log vào file của phiên hiện tại
                    with open(LOG_FILE, "a") as f:
                        f.write(log_entry + "\n")
                    
                    # GỢI Ý: Code lưu ảnh sản phẩm lỗi cho Demo sau này
                    # img_name = f"cam{cam_id+1}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    # cv2.imwrite(os.path.join(SAVE_DIR, img_name), batch_frames[cam_id])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        # In FPS tổng để giám sát tải hệ thống
        print(f"[PERFORMANCE] Total Batch FPS: {fps:.1f}")

if __name__ == "__main__":
    # Khởi chạy các luồng camera
    for i in range(NUM_CAMS):
        t = threading.Thread(target=receive_stream, args=(i, STREAM_URLS[i]))
        t.daemon = True
        t.start()
    
    # Bắt đầu xử lý AI trên luồng chính
    process_batch(ENGINE_MODEL)
