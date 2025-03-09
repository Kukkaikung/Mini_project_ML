# Thai License Plate Recognition System

A computer vision application for detecting and recognizing Thai license plates in videos using YOLOv8 and OCR technologies.

![License Plate Detection Demo](https://github.com/yourusername/thai-license-plate-recognition/raw/main/demo/demo_image.jpg)

## 🌟 Features

- **License Plate Detection**: Uses YOLOv8 model trained specifically for Thai license plates
- **Multiple OCR Options**: Choose between two OCR technologies:
  - **EasyOCR**: Pre-trained OCR solution with Thai language support
  - **Custom CRNN Model**: Specialized Convolutional Recurrent Neural Network for Thai license plate text recognition
- **User-Friendly Web Interface**: Built with Flask, Bootstrap, and JavaScript
- **Real-Time Processing**: Process uploaded videos and see results immediately
- **Thumbnail Gallery**: Automatically collect and display detected license plate images

## 🛠️ System Architecture

The system combines several technologies to create an end-to-end license plate recognition pipeline:

1. **Frontend**: HTML, CSS (Bootstrap 5), JavaScript
2. **Backend**: Flask (Python web framework)
3. **Detection**: YOLOv8 custom-trained model for Thai license plate detection
4. **Recognition**: Dual OCR system (EasyOCR and custom CRNN model)
5. **Image Processing**: OpenCV, PIL for image manipulation

## 📋 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- EasyOCR
- Ultralytics YOLOv8
- PIL (Pillow)
- NumPy
- TorchVision

## 🔧 Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/thai-license-plate-recognition.git
   cd thai-license-plate-recognition
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the model files:
   - Create a `model` directory
   - Download the [YOLOv8 model](link-to-your-model) and place it in the model directory as `new_yolo_patience_model.pt`
   - Download the [CRNN model](link-to-your-model) and place it in the model directory as `new_thai_ocr_model.pth`

5. Download the Thai font:
   - Create a `front` directory
   - Download [TH Sarabun New Bold](link-to-font) and place it in the front directory

6. Update the paths in `app.py` if necessary to match your local file structure

## 🚀 Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Follow the steps in the web interface:
   - Select your preferred OCR model (EasyOCR or CRNN)
   - Upload a video containing Thai license plates
   - View the processed video with detected license plates
   - See the extracted license plate thumbnails at the bottom of the page

## 🔍 OCR Models Explanation

### EasyOCR
- General-purpose OCR engine with multi-language support including Thai
- Uses a combination of CRAFT text detector and CRNN-based text recognizer
- Works out-of-box with reasonable accuracy for various text formats

### Custom CRNN (Convolutional Recurrent Neural Network)
- Specialized model trained specifically for Thai license plates
- Architecture:
  - CNN layers to extract visual features
  - Bidirectional GRU for sequence modeling
  - CTC (Connectionist Temporal Classification) for alignment-free sequence prediction
- Optimized for the specific character set found on Thai license plates

## 📁 Project Structure

```
thai-license-plate-recognition/
├── app.py                     # Main Flask application
├── model/                     # Model directory
│   ├── new_yolo_patience_model.pt  # YOLOv8 model for license plate detection
│   └── new_thai_ocr_model.pth      # CRNN model for OCR
├── front/                     # Fonts directory  
│   └── THSarabunNew Bold.ttf  # Thai font file
├── static/                    # Static files
│   └── uploads/               # Uploaded videos (created at runtime)
├── templates/                 # HTML templates
│   ├── index.html             # Upload page
│   ├── model_selection.html   # OCR model selection page
│   └── video.html             # Results page
└── requirements.txt           # Python dependencies
```

## ⚙️ Model Details

### YOLOv8 License Plate Detection Model
- Custom-trained on Thai license plate dataset
- Detects license plates with minimum dimensions of 49x24 pixels
- Provides bounding box coordinates for each detected plate

### CRNN OCR Model
- Supports the character set: `0123456789กขคฆงจฉชซญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ _`
- Processes license plate images resized to 32x128 pixels
- Uses CTC decoder to handle character repetition issues

## 🤔 Common Issues & Troubleshooting

- **Font not found error**: Ensure the Thai font file is in the correct directory or update the path in `app.py`
- **Model loading errors**: Check if the model files are downloaded and placed in the correct locations
- **Video processing issues**: Ensure the uploaded video is in a standard format (MP4, AVI, etc.)
- **OCR accuracy**: For best results, use videos with clear, well-lit license plates

## 🔮 Future Improvements

- Add support for image uploads in addition to videos
- Implement a database to store detected license plates
- Add a search functionality for historical license plate detections
- Improve OCR accuracy through model fine-tuning
- Support for real-time processing via webcam

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
