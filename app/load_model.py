import torch
import torch.nn as nn
import cv2
import numpy as np

# 📌 โหลดโมเดล
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# 📌 โหลดโมเดลที่เทรนไว้
model = BoundingBoxModel().to(device)
model.load_state_dict(torch.load("C:\\Users\\ASUS\\Downloads\\best_model.pth", map_location=device))
model.eval()



# 📌 เปิดวิดีโอ
cap = cv2.VideoCapture("C:\\Users\\ASUS\\Desktop\\lpr1.mp4")  # หรือใช้ 0 ถ้าเป็นกล้องเว็บแคม

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # ออกจากลูปถ้าวิดีโอจบ
    
    # ✅ Resize และ Normalize ภาพ
    input_img = cv2.resize(frame, (224, 224))
    input_img = input_img / 255.0  # Normalize
    input_tensor = torch.tensor(input_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # ✅ ใช้โมเดลทำนาย
    with torch.no_grad():
        bbox = model(input_tensor).cpu().numpy()[0]  # ได้ค่า (x, y, w, h) ในช่วง [0,1]

    # ✅ แปลง Bounding Box กลับเป็นพิกเซลจริง
    h, w, _ = frame.shape
    x, y, bw, bh = bbox
    x, y, bw, bh = int(x * w), int(y * h), int(bw * w), int(bh * h)

    # ✅ วาด Bounding Box บนเฟรม
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # ✅ แสดงผลวิดีโอ
    cv2.imshow("Object Detection", frame)

    # ✅ กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()