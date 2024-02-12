
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from VisualizationFunctions import visualize_predictions

# FUNCTION TO AUTOMATICALLY TRAIN A MODEL AND PLOT THE RESULTS. IT RETURNS THE LOSSES
def all_in_one(model, train_dataloader, validation_data, epochs, steps_per_epoch, patience, directory, n, pictures):
    
    np.random.seed(42)
    model.compile(optimizer='adam',loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience = patience, 
                                   mode='min', 
                                   verbose=1, 
                                   min_delta=0.0001)

    
    history = model.fit(train_dataloader, 
                        steps_per_epoch = steps_per_epoch, 
                        epochs = 100, 
                        validation_data = validation_data, 
                        validation_batch_size= 32,
                        callbacks = [early_stopping]
                        )
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    min_train_loss = np.amin(loss)
    min_val_loss = np.amin(val_loss)
    
    if pictures:
        visualize_predictions(model, directory, n)
    return min_train_loss, min_val_loss