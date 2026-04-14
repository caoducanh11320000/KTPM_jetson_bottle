import cv2
import threading
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# --- CẤU HÌNH & QUẢN LÝ PHIÊN ---
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"defect_log_{session_id}.txt"
SAVE_DIR = f"defect_images_{session_id}"

MAC_IP = "192.168.2.4"
STREAM_URLS = [f"http://{MAC_IP}:5000/stream/{i}" for i in range(4)]
ENGINE_MODEL = "best.pt"
NUM_CAMS = len(STREAM_URLS)

os.makedirs(SAVE_DIR, exist_ok=True)
with open(LOG_FILE, "w") as f:
    f.write(f"--- PHIÊN LÀM VIỆC: {session_id} ---\nTimestamp,Camera_ID,Status,Confidence\n")

# --- BIẾN TOÀN CỤC CHIA SẺ DỮ LIỆU ---
latest_frames = [None] * NUM_CAMS
frame_locks = [threading.Lock() for _ in range(NUM_CAMS)]
last_detect_time = [0.0] * NUM_CAMS # Chống dội (Debouncing)

# 1. LUỒNG ĐỌC MẠNG (Chạy ngầm - Phụ trách I/O)
def receive_stream(cam_id, stream_url):
    MAX_FAILURES = 30  # Số lần read() thất bại liên tiếp trước khi reconnect
    while True:
        cap = cv2.VideoCapture(stream_url)
        print(f"[CAM {cam_id+1}] Đang kết nối tới {stream_url}...")
        
        if not cap.isOpened():
            print(f"[CAM {cam_id+1}] ❌ Không thể mở stream. Thử lại sau 3 giây...")
            # CƠ CHẾ DỌN DẸP DỮ LIỆU THIU: Xóa ảnh cũ khi không thể kết nối
            with frame_locks[cam_id]:
                latest_frames[cam_id] = None
            time.sleep(3)
            continue
        
        print(f"[CAM {cam_id+1}] ✅ Kết nối thành công!")
        fail_count = 0

        # --- ĐO TỐC ĐỘ NHẬN FRAME TỪ MẠNG ---
        io_frame_count = 0
        io_prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if ret:
                fail_count = 0
                frame = cv2.resize(frame, (480, 480))
                with frame_locks[cam_id]:
                    latest_frames[cam_id] = frame

                # Đếm và in FPS mạng mỗi 3 giây
                io_frame_count += 1
                io_elapsed = time.time() - io_prev_time
                if io_elapsed >= 3.0:
                    io_fps = io_frame_count / io_elapsed
                    print(f"  [NET I/O] Cam_{cam_id+1} nhận được: {io_fps:.1f} FPS từ mạng")
                    io_frame_count = 0
                    io_prev_time = time.time()
            else:
                fail_count += 1
                if fail_count >= MAX_FAILURES:
                    print(f"[CAM {cam_id+1}] ⚠️ Mất tín hiệu ({MAX_FAILURES} lần thất bại). Đang reconnect...")
                    # CƠ CHẾ DỌN DẸP DỮ LIỆU THIU: Xóa ảnh cũ trước khi thoát để reconnect
                    with frame_locks[cam_id]:
                        latest_frames[cam_id] = None
                    break  # Thoát vòng trong → reconnect ở vòng ngoài
                time.sleep(0.1)
        
        cap.release()

# 2. LUỒNG TRUNG TÂM (Chạy AI - Phụ trách Compute)
def process_batch(model_path):
    model = YOLO(model_path, task='detect')
    time.sleep(2)
    prev_time = time.time()
    DEFECT_CLASS_ID = 0

    while True:
        batch_frames = []
        # Gom lô an toàn
        for i in range(NUM_CAMS):
            with frame_locks[i]:
                if latest_frames[i] is not None:
                    batch_frames.append(latest_frames[i].copy())
                else:
                    batch_frames.append(np.zeros((480, 480, 3), dtype=np.uint8))
        
        # Suy luận bằng Tensor Cores
        results = model(batch_frames, verbose=False, half=True)
        curr_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Xử lý Logic Nghiệp vụ (Đếm & Log)
        for cam_id, res in enumerate(results):
            for box in res.boxes:
                if int(box.cls[0]) == DEFECT_CLASS_ID:
                    detect_time = time.time()
                    # Kỹ thuật Cooldown (2 giây)
                    if detect_time - last_detect_time[cam_id] > 2.0:
                        last_detect_time[cam_id] = detect_time
                        
                        log_entry = f"{curr_timestamp},Cam_{cam_id+1},BROKEN,{float(box.conf[0]):.2f}"
                        print(f"⚠️  PHÁT HIỆN LỖI: {log_entry}")
                        with open(LOG_FILE, "a") as f:
                            f.write(log_entry + "\n")
                        
                        # Vẽ bounding box lên ảnh minh chứng rồi lưu
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        evidence_frame = batch_frames[cam_id].copy()
                        # Vẽ hình chữ nhật đỏ
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Vẽ nhãn nền đỏ phía trên box
                        label = f"BROKEN {float(box.conf[0]):.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(evidence_frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 0, 255), -1)
                        cv2.putText(evidence_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Lưu ảnh với Timestamp an toàn cho File System (thay ':' bằng '-')
                        img_name = f"{curr_timestamp.replace(':', '-')}_Cam{cam_id+1}_{float(box.conf[0]):.2f}.jpg"
                        cv2.imwrite(os.path.join(SAVE_DIR, img_name), evidence_frame)

        fps_time = time.time()
        print(f"[PERFORMANCE] Total Batch FPS: {1 / (fps_time - prev_time):.1f}")
        prev_time = fps_time

if __name__ == "__main__":
    for i in range(NUM_CAMS):
        t = threading.Thread(target=receive_stream, args=(i, STREAM_URLS[i]))
        t.daemon = True
        t.start()
    process_batch(ENGINE_MODEL)
