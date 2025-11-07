import uvicorn
import cv2
import numpy as np
import io
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
from ultralytics import YOLO
import csv
from datetime import datetime
import os

DETECTION_LOG_FILE = 'aetherion_detections.csv'

# --- 1. SET YOUR MODEL PATH ---
MODEL_PATH = 'C:/Users/akhin/Downloads/Aetherion_Project/runs/detect/yolov8s_p2_finetune/weights/best.pt'
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO(MODEL_PATH)
    print("--- Aetherion Model Loaded Successfully ---")
    yield
    print("--- Shutting down ---")

app = FastAPI(
    title="Aetherion™ Live Demo System",
    lifespan=lifespan
)

origins = ["null", "*"] # Allows local file and all other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def log_detection(filename: str, confidence: float, bbox: List):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(DETECTION_LOG_FILE)
    with open(DETECTION_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'source_image', 'confidence', 'bounding_box'])
        writer.writerow([timestamp, filename, f"{confidence:.2f}", str(bbox)])

@app.post("/upload_and_detect/")
async def upload_and_detect(file: UploadFile = File(...)):
    global model
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model.predict(img, imgsz=800, conf=0.25)

    bird_found = False
    plotted_image = results[0].plot() 
    plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB) 
    _, buffer = cv2.imencode('.jpg', plotted_image_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    image_data_uri = f"data:image/jpeg;base64,{img_base64}"
    log_messages = [] 

    for r in results:
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            
            if cls == 0: 
                bird_found = True
                bbox = [round(num) for num in box.xyxy[0].tolist()]
                alert_message = f"ALERT: Bird Detected! Confidence: {conf:.2f}, Box: {bbox}"
                log_messages.append(alert_message)
                print(f"🚨 Bird Detected. Alert sent.")
                log_detection(file.filename, conf, bbox)

    if not bird_found:
        log_messages.append("STATUS: Image processed. No birds detected.")
        print("✅ No birds detected.")

    return {
        "filename": file.filename, 
        "bird_detected": bird_found,
        "image_data": image_data_uri,
        "log_messages": log_messages
    }

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("run_demo:app", host="127.0.0.1", port=8000, reload=True)
