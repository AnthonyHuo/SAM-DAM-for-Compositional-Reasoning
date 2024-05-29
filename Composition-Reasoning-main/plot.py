import torch
import cv2
import numpy as np
def depth_to_color(depth, max_depth=1.5, color_format='RGB'):
    """
    Converts a depth value to a color within the red to orange spectrum, 
    dynamically adjusting based on the depth.
    """
    # Normalize the depth value to a 0-1 scale
    print(depth)
    normalized_depth = depth / max_depth
    
    # Map the normalized depth to the red channel (255 to 255, always full in RGB)
    red = 255
    
    # Dynamically adjust the green channel based on depth (0 for red, up to 165 for orange)
    green = int(normalized_depth * 165)
    
    # Blue channel is always 0 for red/orange spectrum
    blue = 0
    
    # Arrange the color values based on the specified format
    if color_format.upper() == 'BGR':
        return [blue, green, red]  # For BGR format
    else:
        return [red, green, blue]  # Default to RGB format

def apply_mask_and_modify(image, mask, depth, color_format='RGB'):
    """
    Fills the masked area of the image with a dynamic color based on the depth value,
    varying within the red to orange spectrum, and accounts for RGB/BGR formats.
    """
    # Ensure mask is boolean for indexing
    mask = mask.astype(bool)
    
    # Convert depth to color, specifying the image's color format
    color = depth_to_color(depth, color_format=color_format)
    
    # Create a copy of the image to modify
    modified_image = image.copy()
    
    # Apply color to the masked area, considering the image format
    for i in range(3):  # Apply each color channel individually
        modified_image[mask, i] = color[i]
    
    return modified_image
# Load the image
image_path = '/home/mhuo/demo5.jpg'  # Specify the path to your image
image = cv2.imread(image_path)

# Load the data from the .pt files
ins_infos = torch.load('/home/mhuo/Composition-Reasoning-main/ins_infos.pt')
ins_infos1 = torch.load('/home/mhuo/Composition-Reasoning-main/ins_infos1.pt')

# Assuming ins_infos and ins_infos1 contain 'mask' and 'depth' keys
mask1 = ins_infos[0][0]['mask'].cpu().numpy()  # Convert the tensor mask to a NumPy array
depth1 = ins_infos[0][0]['depth']

mask2 = ins_infos1[0][0]['mask'].cpu().numpy()
depth2 = ins_infos1[0][0]['depth']

# Apply mask1 and modify the image based on depth1
modified_image1 = apply_mask_and_modify(image, mask1, depth1)

# Apply mask2 and modify the already modified image based on depth2
final_modified_image = apply_mask_and_modify(modified_image1, mask2, depth2)

# Save the final modified image
output_image_path = '/home/mhuo/Composition-Reasoning-main/output_image.jpg'  # Specify the output path
cv2.imwrite(output_image_path, final_modified_image)

print(f"Saved modified image to {output_image_path}")
