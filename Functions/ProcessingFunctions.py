import numpy as np

def preprocess_single_image(image, height, width):
    
    from keras.preprocessing.image import img_to_array
    from skimage.color import rgb2lab
    
    img_shape = image.size
    resized_image = image.resize((height, width))
    resized_array = img_to_array(resized_image)
    image = np.array(resized_array, dtype=float)
    
    X = rgb2lab(1.0/255*image)[:,:,0]/100        #First we transform the input from RGB to LAB, then we normalize
    Y = rgb2lab(1.0/255*image)[:,:,1:]/128       #The normalization is 1/128 for the a and b channels because the input is in [-128,128]

    X = X.reshape(1, height, width, 1)
    Y = Y.reshape(1, height, width, 2)
    return img_shape, X, Y


def preprocess_images(images, height, width):
    X_batch = []
    Y_batch = []
    img_shapes = []
    
    for image in images:
        img_shape, X, Y = preprocess_single_image(image, height, width)
        img_shapes.append(img_shape)
        X_batch.append(X)
        Y_batch.append(Y)
    
    X_batch = np.concatenate(X_batch, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    
    return img_shapes, X_batch, Y_batch


# Function to remove pictures which are black and white
def remove_outliers(X_batch, Y_batch, quantile):
  
    norm_on_input = np.linalg.norm(Y_batch, axis = (1,2))
    norm_on_channels = np.linalg.norm(norm_on_input, axis = 1)
    q = np.quantile(norm_on_channels, quantile)
    temp = X_batch
    temp2 = Y_batch
    for i in reversed(range(len(Y_batch))):
  
        if norm_on_channels[i] <= q:
            temp = np.delete(temp, [i], axis = 0)
            temp2 = np.delete(temp2, [i], axis = 0)
    return temp, temp2