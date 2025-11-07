import torch
from ultralytics import YOLO

def transfer_weights(original_weights_path, custom_yaml_path, output_weights_path):
    print("Initializing custom model with P2 head...")
    custom_model = YOLO(custom_yaml_path)

    print(f"Loading pre-trained weights from {original_weights_path}...")
    checkpoint = torch.load(original_weights_path, map_location=torch.device('cpu'))
    original_state_dict = checkpoint['model'].state_dict()
    custom_state_dict = custom_model.model.state_dict()

    matched_layers = 0
    for name, param in custom_state_dict.items():
        if name in original_state_dict and original_state_dict[name].shape == param.shape:
            param.copy_(original_state_dict[name])
            matched_layers += 1
        else:
            print(f"Skipping layer (mismatch or new): {name}")

    print(f"\nSuccessfully transferred weights for {matched_layers} layers.")
    torch.save({'model': custom_model.model}, output_weights_path)
    print(f"New weight file saved to: {output_weights_path}")

if __name__ == '__main__':
    original_weights = 'yolov8s.pt'  # Assumes yolov8s.pt is in the same folder
    custom_yaml = 'yolov8s_p2.yaml'
    output_weights = 'yolov8s_p2_transfer.pt'
    transfer_weights(original_weights, custom_yaml, output_weights)
