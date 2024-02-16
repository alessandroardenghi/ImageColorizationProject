import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

def augment_directory(image_directory, augmented_directory, height, width):
    
    aug_generator = ImageDataGenerator(
    rotation_range=40,         # Random rotation between -40 and 40 degrees
    width_shift_range=0.2,     # Randomly shift the width by 20%
    height_shift_range=0.2,    # Randomly shift the height by 20%
    shear_range=0.2,           # Shear transformations with intensity of 20%
    zoom_range=0.2,            # Randomly zoom by 20%
    horizontal_flip=True,      # Randomly flip images horizontally
    vertical_flip=False,       # Do not flip images vertically
    fill_mode='nearest'        # Fill any newly created pixels after rotation or shifting
    )
    
    # Directory containing the images
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
        
            image_path = os.path.join(image_directory, filename)

            
            img = load_img(image_path, target_size = (height, width))

            
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Creation of a Directory to store the newly generated images
            augmented_directory = os.path.join(augmented_directory)
            os.makedirs(augmented_directory, exist_ok = True)

            # Generation of augmented images
            aug_iter = aug_generator.flow(x, batch_size = 1, save_to_dir = augmented_directory, save_prefix = 'aug_', save_format = 'jpeg')

            
            num_augmented = 10  # Number of augmented images to generate
            for _ in range(num_augmented):
                augmented_image = next(aug_iter)[0].astype('uint8')

    print("Data augmentation completed.")
    
    
def visualize_augmentation(img_path): 
    
    aug_generator = ImageDataGenerator(
    rotation_range=40,         # Random rotation between -40 and 40 degrees
    width_shift_range=0.2,     # Randomly shift the width by 20%
    height_shift_range=0.2,    # Randomly shift the height by 20%
    shear_range=0.2,           # Shear transformations with intensity of 20%
    zoom_range=0.2,            # Randomly zoom by 20%
    horizontal_flip=True,      # Randomly flip images horizontally
    vertical_flip=False,       # Do not flip images vertically
    fill_mode='nearest'        # Fill any newly created pixels after rotation or shifting
    )

    img = load_img(img_path, target_size=(224, 224))

    # Convert the image to an array
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Generate augmented images
    aug_iter = aug_generator.flow(x, batch_size=1)

    # Generate and preview augmented images
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes = axes.ravel()

    for i in range(5):
        augmented_image = next(aug_iter)[0].astype(np.uint8)
        axes[i].imshow(augmented_image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()