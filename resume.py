from ultralytics import YOLO

if __name__ == '__main__':
    # Path to the LAST checkpoint from your Stage 2 run
    model_path = 'C:/Users/akhin/Downloads/Aetherion_Project/runs/detect/yolov8s_p2_finetune/weights/last.pt'
    
    model = YOLO(model_path)
    
    print("\n--- RESUMING Stage 2 Fine-Tuning ---")
    
    model.train(
        resume=True,
        workers=4
    )
