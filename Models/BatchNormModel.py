from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.layers import BatchNormalization
from keras.models import Sequential



def BatchNormModel(height, width):
    Batch_Norm_Model = Sequential()
    Batch_Norm_Model.add(InputLayer(input_shape=(height, width, 1)))
    Batch_Norm_Model.add(BatchNormalization())

    Batch_Norm_Model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same', strides=2))
    Batch_Norm_Model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(BatchNormalization())

    Batch_Norm_Model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', strides=2))
    Batch_Norm_Model.add(BatchNormalization())

    Batch_Norm_Model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', strides=2))
    Batch_Norm_Model.add(UpSampling2D((2, 2)))

    Batch_Norm_Model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(UpSampling2D((2, 2)))

    Batch_Norm_Model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same'))
    Batch_Norm_Model.add(UpSampling2D((2, 2)))

    Batch_Norm_Model.add(Conv2D(2, (3, 3), activation = 'tanh', padding = 'same'))
    
    return Batch_Norm_Model