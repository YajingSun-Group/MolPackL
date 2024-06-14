import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #use GPU:0 only

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from utils import load_data, data_preprocess, split_data, data_normalize
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

gpus = tf.config.experimental.list_physical_devices(device_type='GPU') 
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True) 

print('Tensorflow Version:',tf.__version__)


def main():

    data0, data1, data2, labels, group = load_data()
    data = data_preprocess(data0, data1, data2)
    train_x, test_x, val_x, train_y, test_y, val_y = split_data(data, labels, group)
    train_x, test_x, val_x = data_normalize(train_x, test_x, val_x, data)


    # define callbacks
    early_stopping = EarlyStopping(
        monitor='val_mae',   
        mode = 'min',
        patience=50,           
        restore_best_weights=True 
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.8,
                                patience=10, min_lr=1e-8)

    # build model
    model = build_model()
    print(model.summary())
    # train model
    print('Training...')
    history = model.fit(train_x,
                    train_y, epochs=1000, batch_size=128,
                    validation_data=(val_x,val_y),
                        callbacks=[early_stopping,reduce_lr],
                        verbose=0
                    )

    _,val_r2,val_mae = model.evaluate(val_x,val_y,verbose=0)
    _,test_r2,test_mae = model.evaluate(test_x,test_y,verbose=0)

    print('/n/nTraining Done!')
    print('***************** Results of the best model *****************')
    print('val_r2:',val_r2,'val_mae:',val_mae)
    print('test_r2:',test_r2,'test_mae:',test_mae)
    print('*************************************************************')

    # save model
    model.save('model_band_gap.keras')
    
if __name__ == '__main__':
    main()




