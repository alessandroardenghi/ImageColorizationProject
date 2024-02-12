import numpy as np
from PIL import Image
import os

# This functions acts as a dataloader, and thus allows us to train models on the whole dataset without loading it all
def data_generator(data_dir, batch_size, height, width):
    
    from keras.preprocessing.image import img_to_array
    from skimage.color import rgb2lab
    from ProcessingFunctions import remove_outliers
    
    np.random.seed(42)
    while True:
        X_batch = []
        Y_batch = []
        valid_extensions = ('.jpg', '.png', '.jpeg')
        
        # Get a batch of image paths
        all_files = os.listdir(data_dir)
        image_files = [file for file in all_files if file.lower().endswith(valid_extensions)]
        batch_paths = np.random.choice(image_files, size = int(batch_size * 1/0.8 + 1))
        

        # Load and preprocess the images
        for image_path in batch_paths:
            image = Image.open(os.path.join(data_dir, image_path))
            image = image.resize((height, width))
            image_array = img_to_array(image)

            # Preprocess the image and separate X and Y
            lab_image = rgb2lab(1.0 / 255 * image_array)
            X = lab_image[:, :, 0] / 100  # L channel as X
            Y = lab_image[:, :, 1:] / 128  # a and b channels as Y

            # Reshape and append to the batch
            X = X.reshape(height, width, 1)
            Y = Y.reshape(height, width, 2)
            X_batch.append(X)
            Y_batch.append(Y)

        # Convert to NumPy arrays and yield the batch
        X_batch = np.array(X_batch)
        Y_batch = np.array(Y_batch)
        X_batch, Y_batch = remove_outliers(X_batch, Y_batch, 0.2)
        yield X_batch, Y_batch
        
        
# Function which allows to easily load images from the directory
def load_images_from_directory(directory, n):    # n = number of images to import
    image_list = []
    count = 0
    
    for file_name in sorted(os.listdir(directory)):
        if count >= n:
            break
        
        if file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(directory, file_name)
            try:
                image = Image.open(image_path)
                image_list.append(image)
                count += 1
            except (IOError, OSError):
                print(f"Error opening image: {file_name}")
    
    return image_list