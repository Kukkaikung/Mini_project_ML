import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms  # เพิ่มการ import ที่นี่


class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),  # 🔥 Dropout 20%

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),  # 🔥 Dropout 30%

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),  # 🔥 Dropout 40%
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 🔥 Dropout 50%
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ฟังก์ชัน preprocess สำหรับเตรียมภาพก่อนนำไปใช้กับโมเดล
def preprocess(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor

# ฟังก์ชัน postprocess สำหรับผลลัพธ์จากโมเดล
def postprocess(output):
    print(f"output.shape: {output.shape}")  # เพิ่มการพิมพ์มิติของ output
    if output.shape[1] == 4:  # ถ้า output มี 4 ค่า (x1, y1, x2, y2)
        bounding_box = output[0].cpu().detach().numpy()  # เอา batch แรกมา
        print(f"Bounding Box (raw): {bounding_box}")  # แสดงค่าของ bounding box

        # ตรวจสอบและแก้ไขค่า bounding box ให้อยู่ในช่วง 0-1
        bounding_box = np.clip(bounding_box, 0, 1)  # ค่าต่ำสุดไม่ต่ำกว่า 0 ค่าสูงสุดไม่เกิน 1
        print(f"Clipped Bounding Box: {bounding_box}")

        return bounding_box  # คืนค่าพิกัด bounding box
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

# ฟังก์ชันเพื่อวาด bounding box บนภาพ
def draw_bounding_box(frame, bounding_box):
    x1, y1, x2, y2 = bounding_box
    height, width, _ = frame.shape
    
    # ตรวจสอบว่า bounding box คำนวณได้ถูกต้อง
    print(f"Original Bounding Box (raw): {(x1, y1, x2, y2)}")
    
    # คูณค่าพิกัดด้วยขนาดของภาพเพื่อแปลงกลับเป็นพิกัดในภาพ
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    # ตรวจสอบพิกัดหลังการแปลง
    print(f"Scaled Bounding Box: {(x1, y1)} -> {(x2, y2)}")
    
    # วาด bounding box บนภาพ
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame



# โหลดโมเดลทั้งหมด (รวมทั้งโครงสร้างและพารามิเตอร์)
model = torch.load("C:\\Users\\ASUS\\Downloads\\new_best_model.pth",map_location=torch.device('cpu') ,weights_only=False)

# เปลี่ยนโมเดลไปใช้กับอุปกรณ์ที่ต้องการ (เช่น GPU)
device = torch.device("cpu")
model.to(device)

# ตั้งโมเดลเป็นโหมด evaluation สำหรับการทำนาย
model.eval()

# ทดสอบโมเดลกับวิดีโอ (ตัวอย่าง)

# ทดสอบการใช้ preprocess และ postprocess กับวิดีโอ
cap = cv2.VideoCapture("C:\\Users\\ASUS\\Desktop\\lpr1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # เตรียมภาพก่อนใช้กับโมเดล
    input_image = preprocess(frame)

    # ใส่โมเดลและทำการทำนาย
    with torch.no_grad():
        output = model(input_image)

    # แสดงผลลัพธ์จากโมเดล
    bounding_box = postprocess(output)
    result = draw_bounding_box(frame, bounding_box)  # วาด bounding box บนภาพ

    # แสดงผลลัพธ์
    cv2.imshow('Output', result)  # แสดงภาพที่ได้จาก postprocess

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()