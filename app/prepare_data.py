import cv2
import os

# กำหนดค่า
video_path = "D:\\licence_plate\\video\\04.mp4"  # เปลี่ยนเป็นชื่อไฟล์วิดีโอของคุณ
output_folder = "D:\\licence_plate\\pic\\video_04"
capture_interval = 1  # วินาที

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(output_folder, exist_ok=True)

# โหลดวิดีโอ
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดวิดีโอได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดวิดีโอได้")
    exit()

# ดึงค่า FPS และคำนวณเฟรมที่ต้องจับ
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = capture_interval * fps

frame_count = 0
image_count = 229

while True:
    ret, frame = cap.read()
    if not ret:
        break  # จบวิดีโอ

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"{image_count:02d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"บันทึก {filename}")
        image_count += 1

    frame_count += 1

cap.release()
print("เสร็จสิ้น!")
