from ultralytics import YOLO


model = YOLO("yolov8s.pt") 

# Train โมเดล
model.train(
    data="dataset.yaml", 
    epochs=50,            
    batch=4,              
    imgsz=640,            
    device="cuda",        
    workers=2,            
    half=True             
)


results = model.val()


model.export(format="onnx")  
