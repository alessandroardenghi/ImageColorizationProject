import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Visual display of some images from the dataset
def display_images(directory):
    image_files = os.listdir(directory)
    images = []
    counter = 0
    for image_file in image_files:
        counter += 1
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        if counter > 16:
            break

    images = np.array(images, dtype = object)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 6))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_xlabel(f'Size = {images[i].shape}')

    plt.tight_layout()
    plt.show()
    
# FUNCTION TO PLOT A SINGLE IMAGE
def plot_image(image):
    fig = plt.Figure(figsize=(12, 4))
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    
# Useful function which automatically displays original image, greyscale image and predicted image

def plot_predicted_image(image, model, height, width):
    from ProcessingFunctions import preprocess_single_image
    from skimage.color import lab2rgb
    img_shape, X, Y = preprocess_single_image(image, height, width)
    
    # Generate predicted values
    output = model.predict(X)
    
    
    # Combine X (grayscale) and predicted values (AB channels)
    lab_image = np.zeros((height, width, 3))
    lab_image[:,:,0] = X[0][:,:,0] * 100
    lab_image[:,:,1:] = output[0] * 120
    
    lab_image = cv2.resize(lab_image, (img_shape[0], img_shape[1]))
    
    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(lab_image[:,:,0], cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')
    
    axes[2].imshow(lab2rgb(lab_image))
    axes[2].set_title('Image with Predicted Values')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
# FUNCTION WHICH ALLOWS TO PLOT THE PREDICTIONS
def visualize_predictions(model, directory, n):
    
    valid_extensions = ('.jpg', '.png', '.jpeg')
    np.random_seed = 42
    
    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith(valid_extensions)]
    batch_paths = np.random.choice(image_files, size = n)
    
    for image_path in batch_paths:
            image = Image.open(os.path.join(directory, image_path))
            plot_predicted_image(image, model, 200, 200)
            

def plot_loss_histograms(x_data, y_data, names):
    
    x = np.arange(len(names))  
    width = 0.35  
    fig, ax = plt.subplots()
    blueshape = ax.bar(x - width/2, x_data, width, label='Train Loss')
    orangeshape = ax.bar(x + width/2, y_data, width, label='Val Loss')
    #ax.set_ylabel('Loss')
    ax.set_title('Model Losses')
    plt.xticks(x, names)
    ax.legend(loc = "best")
    #ax.bar_label(blueshape, padding=3)
    #ax.bar_label(orangeshape, padding=3)
    plt.ylim(0.01, 0.021)
    plt.show()