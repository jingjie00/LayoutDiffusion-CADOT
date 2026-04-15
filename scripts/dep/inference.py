import torch
import argparse
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.util import loopy
from custom import LayoutImageGenerator

# Dictionary mapping object names to indices
object_name_to_idx = {
    "__image__": 0,
    "__null__": 1,
    "small-object": 2,
    "basketball field": 3,
    "building": 4,
    "crosswalk": 5,
    "football field": 6,
    "graveyard": 7,
    "large vehicle": 8,
    "medium vehicle": 9,
    "playground": 10,
    "roundabout": 11,
    "ship": 12,
    "small vehicle": 13,
    "swimming pool": 14,
    "tennis court": 15,
    "train": 16
}

# Create reverse mapping
idx_to_object_name = {v: k for k, v in object_name_to_idx.items()}

def plot_layout(layout, output_path):
    """Plot the layout and save it as an image"""
    plt.figure(figsize=(10, 10))
    
    # Plot each object in the layout
    for obj in layout:
        class_name, x0, y0, x1, y1 = obj
        if class_name == '__image__':
            continue
            
        # Draw rectangle
        width = x1 - x0
        height = y1 - y0
        rect = plt.Rectangle((x0, y0), width, height, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(x0, y0, class_name, color='blue', fontsize=8)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()

def load_sample_from_dataset(config_path, mode='val', sample_index=1):
    """Load a sample layout from the dataset"""
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Create data loader
    data_loader = build_loaders(cfg, mode=mode)
    
    # Get samples until we reach the desired index
    count = 0
    for batch in loopy(data_loader):
        if count == sample_index:
            imgs, meta_data = batch
            break
        count += 1
    
    # Get the first sample from the batch
    obj_bbox = meta_data['obj_bbox'][0].cpu().numpy()
    obj_class = meta_data['obj_class'][0].cpu().numpy()
    is_valid_obj = meta_data['is_valid_obj'][0].cpu().numpy()
    
    # Convert to list format
    layout = []
    for i in range(len(obj_class)):
        if is_valid_obj[i]:
            # Get the class index
            class_idx = int(obj_class[i])
            # Map the index to class name
            if class_idx == 0:
                class_name = '__image__'
            elif class_idx == 1:
                class_name = '__null__'
            else:
                # For COCO dataset, we need to map the index to the actual class name
                # The indices in COCO start from 1, so we subtract 1 to get the correct index
                class_name = idx_to_object_name[class_idx - 1]
            bbox = obj_bbox[i].tolist()
            layout.append([class_name] + bbox)
    
    return layout

def main():
    parser = argparse.ArgumentParser(description="Generate image from layout (either custom or from dataset)")
    parser.add_argument("--config_file", type=str, default='./configs/COCO-stuff_256x256/LayoutDiffusion_cadot.yaml')
    parser.add_argument("--model_path", type=str, default='./log/COCO-stuff_256x256/LayoutDiffusion_cadot/ema_0.9999_0230000.pt')
    parser.add_argument("--output_path", type=str, default="generated_image.png")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--mode", type=str, default="custom", choices=["custom", "dataset"])
    parser.add_argument("--sample_index", type=int, default=105, help="Index of sample to load from dataset (only used if mode=dataset)")
    args = parser.parse_args()

    # Initialize the generator
    generator = LayoutImageGenerator(args.config_file, args.model_path, args.seed)

    if args.mode == "custom":
        # Example custom layout
        layout = [
            ['__image__', 0.0, 0.0, 1.0, 1.0],
            ['playground', 0.4, 0.5791666507720947, 1.0, 1.0],
            ['building', 0.001687500043772161, 0.41574999690055847, 0.3943749964237213, 0.9820416569709778],
            ['crosswalk', 0.7, 0.5, 0.25312501192092896, 0.4791666567325592],
            ['tennis court', 0.5, 0.9, 0.25312501192092896, 0.4791666567325592]
        ]
    else:
        # Load layout from dataset
        layout = load_sample_from_dataset(args.config_file, mode='val', sample_index=args.sample_index)

    print("Using layout:", layout)

    # Save layout visualization
    layout_output_path = args.output_path.replace('.png', '_layout.png')
    plot_layout(layout, layout_output_path)
    print(f"Layout visualization saved to {layout_output_path}")

    # Generate the image
    generated_image = generator.generate(
        layout,
        steps=args.steps,
        output_path=args.output_path
    )
    
    print(f"Generated image saved to {args.output_path}")

if __name__ == "__main__":
    main() 