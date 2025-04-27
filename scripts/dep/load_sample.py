import torch
from omegaconf import OmegaConf
from layout_diffusion.dataset.data_loader import build_loaders
from layout_diffusion.util import loopy
from plot_layout import plot_layout

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

def load_sample_from_dataset(config_path, mode='train', sample_index=1):
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
            class_name = list(object_name_to_idx.keys())[list(object_name_to_idx.values()).index(obj_class[i])]
            bbox = obj_bbox[i].tolist()
            layout.append([class_name] + bbox)
    
    return layout

if __name__ == "__main__":
    config_path = "./configs/COCO-stuff_256x256/LayoutDiffusion_cadot.yaml"
    sample_layout = load_sample_from_dataset(config_path, mode='train', sample_index=1)
    print("Second sample layout from train set:")
    print(sample_layout)
    
    # Plot the layout
    plot_layout(sample_layout, output_path='validation_sample_layout_2.png') 