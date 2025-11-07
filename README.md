# Aetherion
AI-based bird detection and deterrence System
# Aetherion: AI-Based Bird Detection for Airport Safety

This is the official repository for the final-year B.Tech project, "Aetherion." This system is an end-to-end solution for detecting birds on airport runways in real-time to mitigate bird-strike risks.

The project's core challenge was the failure of standard object detection models to detect tiny (sub-10-pixel) and distant birds. This repository contains the code for the custom-architected model, the full training process, and a live deployment demo.

## Key Features

* **Custom Model Architecture:** The standard `YOLOv8s` model was re-architected with a custom **P2 detection head** (`yolov8s_p2.yaml`) to force the model to learn high-resolution features.
* **Two-Stage Training:** A specialized two-stage training process was used to first train the new head and then fine-tune the entire unfrozen model (`train_phase2.py`, `train_stage2.py`).
* **Live Deployment Demo:** A full-stack demo (`run_demo.py`, `dashboard_with_upload.html`) built with **FastAPI** and **WebSockets**.
* **Real-Time Inference:** The backend loads the trained `.pt` model and runs live inference on uploaded images.
* **Instant Alerts:** The **FastAPI** server sends immediate JSON alerts to a web-based dashboard via **WebSockets** if a bird is detected.
* **Event Logging:** All confirmed detections are automatically logged to a timestamped CSV file (`aetherion_detections.csv`) for historical analysis.

## How to Run the Demo

1.  **Install Dependencies:**
    ```bash
    pip install "uvicorn[standard]" fastapi python-multipart ultralytics opencv-python-headless
    ```

2.  **Start the Server:**
    *(Ensure your trained `best.pt` file is in the correct path inside `run_demo.py`)*.
    ```bash
    python -m uvicorn run_demo:app --reload
    ```

3.  **Open the Dashboard:**
    * Open the `dashboard_with_upload.html` file in any web browser.
    * Upload an image to run a live detection.
