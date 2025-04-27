import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

def plot_layout(layout, output_path='layout_visualization.png', image_size=256):
    # Create a blank white image
    img = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Define colors for different object classes
    colors = {
        'building': 'red',
        'crosswalk': 'blue',
        'playground': 'green',
        'small-object': 'purple',
        'basketball field': 'orange',
        'football field': 'yellow',
        'graveyard': 'brown',
        'large vehicle': 'pink',
        'medium vehicle': 'cyan',
        'roundabout': 'magenta',
        'ship': 'gray',
        'small vehicle': 'olive',
        'swimming pool': 'navy',
        'tennis court': 'lime',
        'train': 'teal'
    }
    
    # Draw each bounding box
    for obj in layout:
        class_name = obj[0]
        x0, y0, x1, y1 = obj[1:]
        
        # Convert normalized coordinates to pixel coordinates
        x0 = int(x0 * image_size)
        y0 = int(y0 * image_size)
        x1 = int(x1 * image_size)
        y1 = int(y1 * image_size)
        
        # Get color for this class
        color = colors.get(class_name, 'black')
        
        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Add label
        draw.text((x0, y0-10), class_name, fill=color)
    
    # Save the image
    img.save(output_path)
    print(f"Layout visualization saved to {output_path}")

if __name__ == "__main__":
    # Example layout
    example_layout = [
        ['building', 0.1, 0.1, 0.4, 0.6],
        ['crosswalk', 0.5, 0.7, 0.9, 0.9],
        ['playground', 0.6, 0.1, 0.9, 0.5]
    ]
    
    plot_layout(example_layout) 