# YOLO Object Detection Training Script for VS Code
# Make sure to install required packages first:
# pip install ultralytics matplotlib pandas numpy tqdm pyyaml

import os
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import glob
from tqdm import tqdm
import shutil

def check_gpu():
    """Check if GPU is available and print system info"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Running on CPU")
    print("=" * 40)

def explore_dataset(dataset_path):
    """Explore and analyze the dataset structure"""
    print("Exploring dataset structure...")
    
    # Check dataset structure
    train_images = len(glob.glob(f'{dataset_path}/train/images/*'))
    train_labels = len(glob.glob(f'{dataset_path}/train/labels/*'))
    valid_images = len(glob.glob(f'{dataset_path}/valid/images/*'))
    valid_labels = len(glob.glob(f'{dataset_path}/valid/labels/*'))
    test_images = len(glob.glob(f'{dataset_path}/test/images/*'))
    test_labels = len(glob.glob(f'{dataset_path}/test/labels/*'))
    
    print(f"Train images: {train_images}")
    print(f"Train labels: {train_labels}")
    print(f"Valid images: {valid_images}")
    print(f"Valid labels: {valid_labels}")
    print(f"Test images: {test_images}")
    print(f"Test labels: {test_labels}")
    
    return train_images > 0 and train_labels > 0

def explore_labels(label_path, num_samples=5):
    """Analyze label files and return class distribution"""
    label_files = glob.glob(f'{label_path}/*.txt')
    print(f"\nExamining {num_samples} sample label files from {label_path}:")

    # Initialize class counter
    class_counts = {}

    # Process all label files to count classes
    for label_file in tqdm(label_files, desc="Processing labels"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1

    # Show sample content
    for i, label_file in enumerate(label_files[:num_samples]):
        print(f"\nSample {i+1}: {os.path.basename(label_file)}")
        with open(label_file, 'r') as f:
            content = f.read()
            print(content if content else "Empty file")

    return class_counts

def create_yaml_config(dataset_path, class_names):
    """Create YAML configuration file for the dataset"""
    data_yaml = {
        'path': dataset_path,
        'train': f'{dataset_path}/train/images',
        'val': f'{dataset_path}/valid/images',
        'test': f'{dataset_path}/test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = f'{dataset_path}/data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"YAML file created at {yaml_path}")
    with open(yaml_path, 'r') as f:
        print(f.read())
    
    return yaml_path

def train_model(yaml_path, model_size='n', epochs=100, batch_size=16):
    """Train the YOLO model"""
    print(f"Loading YOLOv8{model_size} model...")
    model = YOLO(f'yolov8{model_size}.pt')
    
    print("Starting model training...")
    
    # Training parameters optimized for performance
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        patience=15,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        nbs=64,
        overlap_mask=True,
        close_mosaic=10,
        amp=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True,
        
        # Data augmentation parameters
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    
    return model, results

def evaluate_model(model, class_names):
    """Evaluate the trained model"""
    print("\nModel training complete. Running validation...")
    val_results = model.val()

    # Print metrics
    print("\nValidation Results:")
    metrics = val_results.box
    print(f"Precision: {metrics.mp:.4f}")
    print(f"Recall: {metrics.mr:.4f}")
    print(f"mAP50: {metrics.map50:.4f}")
    print(f"mAP50-95: {metrics.map:.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    for i, cls_name in enumerate(class_names):
        if hasattr(metrics, 'ap_class_index') and i < len(metrics.ap_class_index):
            idx = np.where(metrics.ap_class_index == i)[0]
            if len(idx) > 0:
                ap50 = metrics.ap50[idx[0]]
                print(f"Class {i} ({cls_name}): AP50 = {ap50:.4f}")
            else:
                print(f"Class {i} ({cls_name}): No detections")
        else:
            print(f"Class {i} ({cls_name}): No detections")
    
    return val_results

def test_model(model, dataset_path, class_names, num_examples=5):
    """Test the model on test data and visualize results"""
    print("\nRunning inference on test data...")
    test_results = model.predict(
        source=f'{dataset_path}/test/images',
        save=True,
        conf=0.25,
        iou=0.45,
        max_det=300,
        save_conf=True,
        save_txt=True
    )

    # Visualize some predictions
    num_test_examples = min(num_examples, len(test_results))
    print(f"\nVisualizing {num_test_examples} test predictions...")

    for i in range(num_test_examples):
        result = test_results[i]
        img = result.orig_img
        boxes = result.boxes

        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        # Draw bounding boxes and labels
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_name = class_names[cls] if cls < len(class_names) else f"Unknown-{cls}"
                label = f"{cls_name} {conf:.2f}"

                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                                 fill=False, color='red', linewidth=2))
                plt.text(x1, y1, label, color='white', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))

        plt.axis('off')
        plt.title(f'Test Image {i+1}')
        plt.tight_layout()
        plt.show()
    
    return test_results

def export_model(model, export_formats=['onnx', 'torchscript']):
    """Export model to different formats for deployment"""
    print("\nOptimizing model for deployment...")
    exported_models = {}
    
    for format_name in export_formats:
        try:
            if format_name == 'onnx':
                path = model.export(format='onnx', dynamic=True, simplify=True)
                exported_models['onnx'] = path
                print(f"Model exported to ONNX: {path}")
            elif format_name == 'torchscript':
                path = model.export(format='torchscript')
                exported_models['torchscript'] = path
                print(f"Model exported to TorchScript: {path}")
            elif format_name == 'tflite':
                path = model.export(format='tflite')
                exported_models['tflite'] = path
                print(f"Model exported to TFLite: {path}")
        except Exception as e:
            print(f"Failed to export to {format_name}: {e}")
    
    return exported_models

def benchmark_model(model_path, test_image_path, num_inferences=50):
    """Benchmark model inference speed"""
    print("\nMeasuring inference speed...")
    
    model = YOLO(model_path)
    
    # Warmup
    print("Warming up model...")
    for _ in range(10):
        _ = model(test_image_path)

    # Benchmark
    print("Benchmarking inference time...")
    times = []

    for _ in tqdm(range(num_inferences), desc="Running inference"):
        start = time.time()
        _ = model(test_image_path)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {1/avg_time:.2f}")

    # Plot inference time distribution
    plt.figure(figsize=(10, 5))
    plt.hist(np.array(times) * 1000, bins=20)
    plt.title('Inference Time Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
    return avg_time

def analyze_class_performance(val_results, class_names, precision_threshold=0.7):
    """Analyze per-class performance and identify low-performing classes"""
    print("\nAnalyzing per-class performance...")
    
    if hasattr(val_results.box, 'ap_class_index') and hasattr(val_results.box, 'ap50'):
        low_precision_classes = []
        
        for i, idx in enumerate(val_results.box.ap_class_index):
            ap50 = val_results.box.ap50[i]
            if ap50 < precision_threshold:
                class_name = class_names[idx] if idx < len(class_names) else f"Unknown-{idx}"
                low_precision_classes.append((class_name, ap50))

        print(f"Classes with low precision (AP50 < {precision_threshold}):")
        for cls, prec in low_precision_classes:
            print(f"{cls}: {prec:.4f}")

        if not low_precision_classes:
            print("All classes have good precision!")
    else:
        print("Detailed per-class metrics not available")

def save_model(model, save_directory="./trained_models"):
    """Save the trained model"""
    print(f"\nSaving model to {save_directory}...")
    os.makedirs(save_directory, exist_ok=True)
    
    # Get the best model path
    best_model_path = model.ckpt_path if hasattr(model, 'ckpt_path') else 'runs/detect/train/weights/best.pt'
    
    # Copy the best model
    if os.path.exists(best_model_path):
        destination = os.path.join(save_directory, 'best_model.pt')
        shutil.copy(best_model_path, destination)
        print(f"Best model saved to {destination}")
        return destination
    else:
        print("Best model path not found!")
        return None

def main():
    """Main function to run the complete YOLO training pipeline"""
    
    # Configuration - Update these paths according to your setup
    DATASET_PATH = "./dataset"  # Your dataset folder path
    MODEL_SIZE = 'n'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
    EPOCHS = 100
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    
    # Class names for your dataset (you'll need to update this based on your actual classes)
    CLASS_NAMES = [
        'Bag', 'Chair', 'Cup-mug', 'Door', 'Eye-glass', 'Fridge', 'Handbag', 
        'Keyboard', 'Laptop', 'Phone', 'Plastic bottle', 'Spoon', 'TV', 'Wallet', 
        'Watch', 'book', 'bowl', 'box', 'chocolate', 'coin', 'couch', 
        'fire-extinguisher', 'flowerPot', 'glass-tumbler', 'headphones', 'helmet', 
        'jars', 'knife', 'mouse', 'plate', 'router', 'scissor', 'shoe', 'speaker'
    ]
    
    print("Starting YOLO Object Detection Training Pipeline")
    print("=" * 50)
    
    # Step 1: Check system requirements
    check_gpu()
    
    # Step 2: Explore dataset
    if not explore_dataset(DATASET_PATH):
        print("Dataset exploration failed. Please check your dataset path and structure.")
        return
    
    # Step 3: Analyze labels and class distribution
    train_class_counts = explore_labels(f'{DATASET_PATH}/train/labels')
    
    print("\nClass distribution in training set:")
    for class_id, count in sorted(train_class_counts.items()):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown-{class_id}"
        print(f"Class {class_id} ({class_name}): {count} instances")
    
    # Visualize class distribution
    plt.figure(figsize=(15, 6))
    classes = list(train_class_counts.keys())
    counts = list(train_class_counts.values())
    plt.bar(classes, counts)
    plt.title('Class Distribution in Training Dataset')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.xticks(classes)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Step 4: Create or update YAML configuration
    yaml_path = create_yaml_config(DATASET_PATH, CLASS_NAMES)
    
    # Step 5: Train the model
    model, training_results = train_model(yaml_path, MODEL_SIZE, EPOCHS, BATCH_SIZE)
    
    # Step 6: Evaluate the model
    val_results = evaluate_model(model, CLASS_NAMES)
    
    # Step 7: Test on unseen data
    test_results = test_model(model, DATASET_PATH, CLASS_NAMES)
    
    # Step 8: Export model for deployment
    exported_models = export_model(model)
    
    # Step 9: Benchmark inference speed
    test_images = glob.glob(f'{DATASET_PATH}/test/images/*')
    if test_images and 'onnx' in exported_models:
        benchmark_model(exported_models['onnx'], test_images[0])
    elif test_images:
        benchmark_model(model.ckpt_path, test_images[0])
    
    # Step 10: Analyze class performance
    analyze_class_performance(val_results, CLASS_NAMES)
    
    # Step 11: Save the final model
    saved_model_path = save_model(model)
    
    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print(f"Best model saved at: {saved_model_path}")
    print("Check the 'runs/detect/train' folder for detailed training logs and visualizations.")

if __name__ == "__main__":
    main()
