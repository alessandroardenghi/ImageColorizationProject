from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential

def BasicModel(height, width):
    Basic_Model = Sequential()
    Basic_Model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', input_shape = (height, width, 1)))
    Basic_Model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    Basic_Model.add(Conv2D(2, (3, 3), activation = 'tanh', padding = 'same'))
    return Basic_Model