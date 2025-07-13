import os
import numpy as np
from PIL import Image

# Define paths
cat_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\resized_cat'
dog_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\resized_dog'
volume_cat_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\volumes_cats'
volume_dog_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\volumes_dogs'

# Create directories for volumes if they don't exist
os.makedirs(volume_cat_folder, exist_ok=True)
os.makedirs(volume_dog_folder, exist_ok=True)

# Function to stack images into 3D volumes
def create_volumes(input_folder, output_folder, volume_size=10):
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff'))])
    num_images = len(image_files)
    
    for i in range(0, num_images - volume_size + 1, volume_size):
        volume = []
        for j in range(volume_size):
            img_path = os.path.join(input_folder, image_files[i + j])
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    
                    # Check if the shape is consistent with the first image
                    if volume and img_array.shape != volume[0].shape:
                        print(f"Skipping {image_files[i + j]} due to shape mismatch: {img_array.shape}")
                        continue
                    
                    volume.append(img_array)
                    
            except Exception as e:
                print(f"Error processing {image_files[i + j]}: {e}")
                continue

        # Ensure all images have the same shape before stacking
        if len(volume) == volume_size:
            volume = np.stack(volume, axis=0)  # Shape: (volume_size, height, width, channels)
            
            # Save volume as .npy file
            volume_filename = f'volume_{i // volume_size:04d}.npy'
            volume_path = os.path.join(output_folder, volume_filename)
            np.save(volume_path, volume)
            print(f"Saved volume: {volume_filename}")
        else:
            print(f"Volume starting at index {i} does not have the correct number of images and was skipped.")

# Create volumes for cats and dogs
create_volumes(cat_folder, volume_cat_folder)
create_volumes(dog_folder, volume_dog_folder)
