import os
import cv2
import easyocr
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, session, redirect, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import base64
from torchvision import transforms

app = Flask(__name__)
app.secret_key = os.urandom(24)

# โหลดโมเดล YOLOv8
model = YOLO("C://Users//ASUS//Desktop//Mini_project_ML//model//new_yolo_patience_model.pt")

# โหลด EasyOCR
reader = easyocr.Reader(['en', 'th'])

# โหลดฟอนต์ภาษาไทย
font_path = "C://Users//ASUS//Desktop//Mini_project_ML//front//THSarabunNew Bold.ttf"
if not os.path.exists(font_path):
    print("❌ Thai font not found!")

# เงื่อนไขขนาดป้ายทะเบียน
MIN_WIDTH, MIN_HEIGHT, MIN_AREA = 49, 24, 1176

# ✅ แก้ไข characters ให้ตรงกับโมเดล
characters = "0123456789กขคฆงจฉชซญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ _"


plate_thumbnails = []

# โมเดล CRNN สำหรับ OCR
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )
        self.rnn = nn.GRU(1024, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=2)

# โหลดโมเดล CRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_crnn = CRNN(num_classes=len(characters) + 1)

try:
    model_crnn.load_state_dict(torch.load('C://Users//ASUS//Desktop//Mini_project_ML//model//new_thai_ocr_model.pth', map_location=device))
    print("✅ CRNN Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CRNN model: {e}")

model_crnn = model_crnn.to(device)
model_crnn.eval()

# Transform สำหรับ OCR
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ฟังก์ชันวาดข้อความ
def draw_text_thai(image, text, position, font_size=32, color=(0, 255, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# ✅ ฟังก์ชันถอดรหัส CTC (แก้ปัญหาตัวซ้ำ)
def decode_predictions(preds):
    pred_str = []
    last_char = None
    for p in preds:
        if p != last_char and p < len(characters):
            pred_str.append(characters[p])
        last_char = p
    return "".join(pred_str).strip()

# ✅ ฟังก์ชันทำนายด้วย CRNN OCR
def predict(img_pil):
    print("🔍 Pre-processing OCR input...")
    img = transform(img_pil).unsqueeze(0).to(device)
    print("✅ Image shape:", img.shape)  # ตรวจสอบขนาดภาพก่อนเข้าโมเดล
    
    with torch.no_grad():
        outputs = model_crnn(img)
    
    print("📊 CRNN output shape:", outputs.shape)  # ควรเป็น (1, w, num_classes)
    preds = torch.argmax(outputs, dim=2)
    preds = preds.squeeze(0).tolist()
    
    print("📝 Raw Predictions (Converted):", preds)
    
    pred_text = decode_predictions(preds)
    print(f"🔍 CRNN Prediction: {pred_text}")
    return pred_text

# ฟังก์ชันประมวลผลวิดีโอ
def process_video(video_path, ocr_model="easyocr"):
    global plate_thumbnails 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดวิดีโอได้!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        license_plate_images = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) >= MIN_WIDTH and (y2 - y1) >= MIN_HEIGHT:
                    plate_img = frame[y1:y2, x1:x2]
                    if ocr_model == "yoloocr":
                        plate_id = predict(Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)))
                    else:
                        plate_id = " ".join(reader.readtext(plate_img, detail=0))
                    frame = draw_text_thai(frame, plate_id, (x1, y2 + 10))
                    license_plate_images.append(plate_img)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        plate_thumbnails = [encode_image_to_base64(img) for img in license_plate_images]
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


@app.route('/')
def index():
    return redirect(url_for('model_selection'))

@app.route('/model_selection', methods=['GET', 'POST'])
def model_selection():
    if request.method == 'POST':
        session['ocr_model'] = request.form['ocr_model']
        return redirect(url_for('upload_video'))
    return render_template('model_selection.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename:
            os.makedirs('static/uploads', exist_ok=True)
            video_path = os.path.join("static/uploads", file.filename)
            file.save(video_path)
            return render_template('video.html', video_path=video_path, ocr_model=session.get('ocr_model', 'easyocr'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(request.args.get("video_path"), session.get('ocr_model', 'easyocr')), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_plate_thumbnails')
def get_plate_thumbnails():
    global plate_thumbnails  
    return jsonify(plate_thumbnails)

if __name__ == '__main__':
    app.run(debug=True)
