import argparse
import os
import torch
from ultralytics import YOLO

def check_hardware_device(requested_device):
    """Checks for hardware acceleration and returns the optimal device string."""
    if requested_device is not None and requested_device.lower() != "auto":
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Check: Found {torch.cuda.device_count()} GPU(s) -> [{gpu_name}]. Using CUDA.")
        return "0" 
    elif torch.backends.mps.is_available():
        print("Hardware Check: Found Apple Silicon. Using MPS.")
        return "mps"
    else:
        print("Hardware Check: No GPU detected. Falling back to CPU.")
        return "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Universal Instance Segmentation Training Pipeline for EM Images")

    # Core required arguments - UPDATED FOR SEGMENTATION
    parser.add_argument("--model", type=str, default="yolov8m-seg.pt", help="Model architecture (use -seg.pt for segmentation)")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to the dataset YAML file")
    
    # Standard Ultralytics Parameters - UPDATED FOR HIGH-RES EM IMAGES
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution (crucial for capturing small organelles)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (reduce if out of RAM)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default= "AdamW", choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'], help="Optimizer algorithm")
    parser.add_argument("--amp", type=bool, default=None, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")

    # Custom Parameters - UPDATED FOR BIOLOGICAL STRUCTURES
    parser.add_argument("--fl_gamma", type=float, default=0.0, help="Focal Loss gamma for class imbalance (scale of organelles)")
    parser.add_argument("--degrees", type=float, default=90.0, help="Image rotation augmentation (EM images are rotation-invariant)")
    parser.add_argument("--flipud", type=float, default=0.5, help="Flip up-down probability")
    parser.add_argument("--device", type=str, default="auto", help="Hardware override: '0', '0,1', 'cpu', 'mps', or 'auto'")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Extract the clean model name for output directories
    model_base_name = os.path.splitext(os.path.basename(args.model))[0]
    target_project = "Result"
    target_name = model_base_name

    print("="*60)
    print("Initializing High-Res EM Segmentation Pipeline")
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Output:  {target_project}/{target_name}")
    print("="*60)

    optimal_device = check_hardware_device(args.device)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"CRITICAL ERROR: Dataset file '{args.data}' not found.")

    try:
        model = YOLO(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{args.model}'. Error: {e}")

    # Base kwargs that are always required or dynamically computed by our script
    training_kwargs = {
        'data': args.data,
        'project': target_project,
        'name': target_name,
        'device': optimal_device,
    }

    # Standard YOLO Arguments Injection
    # Added 'degrees' and 'flipud' to the list to ensure they are passed to the trainer
    standard_args = ['epochs', 'batch', 'imgsz', 'patience', 'workers', 'lr0', 'optimizer', 'amp', 'degrees', 'flipud']
    
    for arg in standard_args:
        value = getattr(args, arg)
        if value is not None:
            training_kwargs[arg] = value
            print(f"Parameter Override: {arg} = {value}")

    # Handle Special Flags
    if args.resume:
        training_kwargs['resume'] = True
        print("Parameter Override: Resuming training from checkpoint.")

    if args.fl_gamma > 0.0:
        training_kwargs['fl_gamma'] = args.fl_gamma
        print(f"Focal Loss Enabled (gamma={args.fl_gamma}).")

    # Execute Training
    try:
        print("\nStarting training loop...")
        results = model.train(**training_kwargs)
        print(f"\nTraining complete! Check the '{target_project}' directory for results.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed due to an error: {e}")

if __name__ == "__main__":
    main()
    

# Execute the script in terminal with the following command to start training with the specified parameters: 
"""
python models/yolo.py --model yolov8m-seg.pt --data data.yaml --epochs 100 
--batch 16 --imgsz 1024 --patience 10 --workers 4 --lr0 0.001 
--optimizer AdamW --amp True --fl_gamma 1.5 --degrees 90.0 --flipud 0.5 --device auto
"""

# Simple Scripts
"""
python train.py --model yolov8m-seg.pt --data data.yaml --epochs 100 --batch 8 --device auto
"""