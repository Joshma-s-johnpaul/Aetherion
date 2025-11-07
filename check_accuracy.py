from ultralytics import YOLO

if __name__ == '__main__':
    # --- 1. SET PATHS ---
    # Path to the model you want to test
    model_path = 'C:/Users/akhin/Downloads/Aetherion_Project/runs/detect/yolov8s_p2_finetune/weights/best.pt'
    
    # Path to your dataset's YAML file
    data_path = 'C:/Users/akhin/Downloads/Aetherion_Project/airbirds_dataset.yaml'

    # --- 2. LOAD THE MODEL ---
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded. Starting validation...")

    # --- 3. RUN VALIDATION ---
    metrics = model.val(
        data=data_path,
        imgsz=800,
        batch=4,
        split='val', 
        name='finetune_accuracy_check'
    )

    print("\n--- Validation Complete ---")
    print(f"Metrics saved to 'runs/detect/finetune_accuracy_check'")
    print("\n--- Key Accuracy Metrics ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f" mAP75: {metrics.box.map75:.4f}")
