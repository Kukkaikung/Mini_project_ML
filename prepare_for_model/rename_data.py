import os
import glob

# ตั้งค่า path หลักที่มีโฟลเดอร์เก็บรูปภาพ
base_path = "D:\\licence_plate\\pic"  # เปลี่ยนเป็น path ของคุณ

counter = 1  # เริ่มต้นนับเลขไฟล์จาก 1

# วนลูปผ่านแต่ละโฟลเดอร์ย่อยใน base_path (เรียงลำดับชื่อโฟลเดอร์ก่อน)
for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)

    # ตรวจสอบว่าเป็นโฟลเดอร์จริง ๆ
    if os.path.isdir(folder_path):
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))  # ดึงไฟล์รูปทั้งหมด

        # วนลูปเปลี่ยนชื่อไฟล์
        for file_path in image_files:
            ext = os.path.splitext(file_path)[1]  # ดึงนามสกุลไฟล์
            new_name = f"{counter:02d}{ext}"  # ตั้งชื่อใหม่เป็น 01, 02, 03, ...
            new_path = os.path.join(folder_path, new_name)

            os.rename(file_path, new_path)  # เปลี่ยนชื่อไฟล์
            print(f"Renamed: {file_path} -> {new_path}")

            counter += 1  # เพิ่มค่าลำดับไฟล์

print("เปลี่ยนชื่อไฟล์เสร็จสิ้น")

