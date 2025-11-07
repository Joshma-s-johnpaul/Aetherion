from ultralytics import YOLO

def main():
    print("--- Starting Stage 1: Training the new head (freezing backbone) ---")
    
    # Load the model with the newly created transfer weights
    model = YOLO('yolov8s_p2_transfer.pt')

    model.train(
        data='airbirds_dataset.yaml',
        imgsz=800,
        batch=4,
        epochs=25,
        patience=15,
        freeze=10,  # Freezes the backbone (first 10 layers)
        name='yolov8s_p2_stage1',
        workers=4
    )

if __name__ == '__main__':
    main()
