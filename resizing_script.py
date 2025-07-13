import os
from PIL import Image

# Define paths for cat and dog images
cat_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\Cat'
dog_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\Dog'
resized_cat_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\rezised_cat'
resized_dog_folder = r'E:\Priya Research\F Drive\Priya Research\IEEE Data Port\Cats and Dogs\PetImages\resized_dog'

# Create directories for resized images if they don't exist
os.makedirs(resized_cat_folder, exist_ok=True)
os.makedirs(resized_dog_folder, exist_ok=True)

# Resize function
def resize_images(input_folder, output_folder, target_size=(128, 128)):
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        
        try:
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                img.save(os.path.join(output_folder, filename))  # Save resized image
            print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Error resizing {filename}: {e}")

# Resize cat and dog images
resize_images(cat_folder, resized_cat_folder)
resize_images(dog_folder, resized_dog_folder)
