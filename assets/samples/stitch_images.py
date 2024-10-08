from PIL import Image
import os

def resize_image(image, max_width, max_height):
    """
    Resize an image to fit within the specified dimensions while maintaining the aspect ratio.
    
    Parameters:
    - image: A PIL Image object to be resized.
    - max_width: Maximum width of the resized image.
    - max_height: Maximum height of the resized image.
    
    Returns:
    - A resized PIL Image object.
    """
    width, height = image.size
    aspect_ratio = width / height
    
    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    
    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)
    
    return image.resize((width, height), Image.LANCZOS)

def stitch_images(paths, orientation='horizontal', max_width=None, max_height=None):
    """
    Stitch images from given local paths into a single image with optional scaling.
    
    Parameters:
    - paths: List of local image file paths.
    - orientation: 'horizontal' or 'vertical', determines how to concatenate images.
    - max_width: Maximum width of each image after resizing.
    - max_height: Maximum height of each image after resizing.
    
    Returns:
    - A PIL Image object of the stitched image.
    """
    images = []
    
    # Load and resize images from local paths
    for path in paths:
        img = Image.open(path)
        
        if max_width and max_height:
            img = resize_image(img, max_width, max_height)
        
        images.append(img)
    
    # Determine the size of the resulting image
    if orientation == 'horizontal':
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        new_img = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
    
    elif orientation == 'vertical':
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        new_img = Image.new('RGB', (max_width, total_height))
        
        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

    else:
        raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")

    return new_img

root = '/home/cxnam/Documents/MyWorkingSpace/Trainer/inference/samples/a2lm'
file_paths = sorted([file for file in os.listdir(root) if file.startswith("")])
# root = '/home/cxnam/Documents/MEAD/M003/images/front_happy_level_1/001'
# file_paths = sorted([file for file in os.listdir(root) if file.startswith("")])
# root = '/home/cxnam/Documents/MyWorkingSpace/Trainer/inference/samples/lm2face'
# file_paths = sorted([file for file in os.listdir(root) if file.startswith("pred_")])

paths = []
for i in range(15):
    path = os.path.join(root, file_paths[i])
    paths.append(path)

# Create a horizontal stitch with scaling
stitched_image = stitch_images(paths, orientation='horizontal', max_width=256*3, max_height=256*3)

# Save the stitched image
stitched_image.save('/home/cxnam/Documents/MyWorkingSpace/Trainer/assets/samples/M003_happy_1_001_vae_lm_1.jpg')
