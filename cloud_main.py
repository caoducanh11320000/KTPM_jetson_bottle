import cv2
import threading
import time
import numpy as np
import os
import sqlite3
import uuid
from datetime import datetime
import requests
from ultralytics import YOLO


# --- EDGE COMPUTING TELEMETRY CONFIGURATION ---
# Endpoint URL for Microsoft Power Automate Webhook
POWER_AUTOMATE_WEBHOOK = "https://7c5a9ddb6c4ae373b74f66f45050ef.86.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/62014a968e5f4139b4804a2a101eb9c8/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=vKr5lhJTXhNyuC1oGbjB7huXV1yFcyU7pA9lr5lHrHk"
# Cooldown interval (seconds) to prevent burst spam per camera
COOLDOWN_SECONDS = 2.0


# --- CẤU HÌNH & QUẢN LÝ PHIÊN ---
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"defect_log_{session_id}.txt"
SAVE_DIR = f"defect_images_{session_id}"


NODE_ID = "JETSON_LINE_01_HANOI"
DB_FILE = "local_buffer.db"


# --- SQLITE DB INITIALIZATION ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS defect_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  camera_id TEXT,
                  status TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()


# Cấu hình kết nối
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


# --- CLOUD: SQLite Buffer Functions ---
def push_to_db(timestamp, camera_id, status, confidence):
    """ Persist defect telemetry securely to the local SQLite database """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO defect_logs (timestamp, camera_id, status, confidence) VALUES (?, ?, ?, ?)",
              (timestamp, camera_id, status, confidence))
    conn.commit()
    conn.close()


def cloud_sync_worker():
    """ Background daemon responsible for reliable cloud synchronization """
    print("[INFO] Initializing Cloud Sync Worker (SQLite + Chunk Batching)...")
    last_sync_time = time.time()
   
    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
           
            # Retrieve up to 20 oldest unsynced records
            c.execute("SELECT id, timestamp, camera_id, status, confidence FROM defect_logs ORDER BY id ASC LIMIT 20")
            rows = c.fetchall()
           
            current_time = time.time()
            time_since_last_sync = current_time - last_sync_time
           
            # --- HYBRID SYNC TRIGGERS ---
            # Trigger 1: Batch limit reached (20 items)
            # Trigger 2: Time limit reached (5s since last sync) with at least 1 pending item
            # Trigger 3: Heartbeat interval (5 minutes) reached for Sharepoint EdgeNodes status
            need_to_send_defects = len(rows) == 20 or (len(rows) > 0 and time_since_last_sync >= 5)
            need_to_send_ping = time_since_last_sync >= 300
           
            if need_to_send_defects or need_to_send_ping:
                batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:4]}"
                curr_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               
                payload = {
                    "TraceInfo": {
                        "BatchID": batch_id,
                        "NodeID": NODE_ID,
                        "HeartbeatTimestamp": curr_timestamp
                    },
                    "Defects": []
                }
               
                row_ids_to_delete = []
                for r in rows:
                    row_ids_to_delete.append(r[0])
                    payload["Defects"].append({
                        "Timestamp": r[1],
                        "CameraID": r[2],
                        "Status": r[3],
                        "Confidence": r[4]
                    })
               
                res = requests.post(POWER_AUTOMATE_WEBHOOK, json=payload, timeout=8)
               
                if res.ok:
                    if len(rows) > 0:
                        print(f"[INFO] Successfully synced batch ID {batch_id} containing {len(rows)} defects. Executing DB cleanup.")
                        # Purge synchronized records
                        placeholders = ','.join(['?']*len(row_ids_to_delete))
                        c.execute(f"DELETE FROM defect_logs WHERE id IN ({placeholders})", row_ids_to_delete)
                        conn.commit()
                    else:
                        print(f"[INFO] Heartbeat ping (EdgeNode verification) successfully transmitted.")
                   
                    last_sync_time = time.time()
                else:
                    print(f"[WARNING] Cloud connection rejected (Status: {res.status_code}). Buffer retained safely.")
                    time.sleep(5)
           
            conn.close()
            time.sleep(1) # Prevent CPU idle overhead
           
        except Exception as e:
            print(f"[ERROR] Sync worker encountered exception: {e}")
            time.sleep(5)


# 1. LUỒNG ĐỌC MẠNG (Chạy ngầm - Phụ trách I/O)
def receive_stream(cam_id, stream_url):
    MAX_FAILURES = 30  # Số lần read() thất bại liên tiếp trước khi reconnect
    while True:
        cap = cv2.VideoCapture(stream_url)
        print(f"[CAM {cam_id+1}] Đang kết nối tới {stream_url}...")
       
        if not cap.isOpened():
            print(f"[CAM {cam_id+1}] ❌ Không thể mở stream. Thử lại sau 3 giây...")
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
                cls = int(box.cls[0])
                conf = float(box.conf[0])


                if cls == DEFECT_CLASS_ID:
                    detect_time = time.time()
                    # Kỹ thuật Cooldown
                    if detect_time - last_detect_time[cam_id] > COOLDOWN_SECONDS:
                        last_detect_time[cam_id] = detect_time


                        # Dynamically extract labels from PyTorch weights
                        status_name = model.names[cls].upper().replace(" ", "_") if model.names else f"DEFECT_{cls}"
                        log_entry = f"{curr_timestamp},Cam_{cam_id+1},{status_name},{conf:.2f}"
                        print(f"⚠️  PHÁT HIỆN LỖI: {log_entry}")
                        with open(LOG_FILE, "a") as f:
                            f.write(log_entry + "\n")
                       
                        # Forward payload to SQLite buffer → Cloud sync
                        push_to_db(curr_timestamp, f"Cam_{cam_id+1}", status_name, round(conf, 2))


                        # Vẽ bounding box lên ảnh minh chứng rồi lưu
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        evidence_frame = batch_frames[cam_id].copy()
                        # Vẽ hình chữ nhật đỏ
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Vẽ nhãn nền đỏ phía trên box
                        label = f"{status_name} {conf:.2f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(evidence_frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 0, 255), -1)
                        cv2.putText(evidence_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        # Lưu ảnh
                        img_name = f"{curr_timestamp.replace(':', '-')}_Cam{cam_id+1}_{conf:.2f}.jpg"
                        cv2.imwrite(os.path.join(SAVE_DIR, img_name), evidence_frame)


        fps_time = time.time()
        print(f"[PERFORMANCE] Total Batch FPS: {1 / (fps_time - prev_time):.1f}")
        prev_time = fps_time


if __name__ == "__main__":
    init_db()
   
    # Khởi động Cloud Sync Worker (daemon)
    sync_thread = threading.Thread(target=cloud_sync_worker, daemon=True)
    sync_thread.start()


    # Khởi động 4 luồng I/O đọc stream
    for i in range(NUM_CAMS):
        t = threading.Thread(target=receive_stream, args=(i, STREAM_URLS[i]))
        t.daemon = True
        t.start()
   
    # Chạy vòng lặp AI chính
    process_batch(ENGINE_MODEL)
