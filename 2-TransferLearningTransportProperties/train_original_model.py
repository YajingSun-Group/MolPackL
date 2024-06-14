#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train_original_model.py
@Time    :   2024/06/14 13:45:50
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #use GPU:0 only

import pandas as pd
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices(device_type='GPU') 
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True) 

#print GPU Devices
print('Tensorflow Version:',tf.__version__)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import load_data, data_preprocess, split_data, data_normalize
from model import build_model


def main():
    target_label_list = ['density','elec_max_u','elec_trans','hole_max_u','hole_trans']
    for target_label in target_label_list:
        data0, data1, data2, labels, group, cod_id = load_data(target_label)
        data = data_preprocess(data0, data1, data2)
        train_x, test_x, val_x, train_y, test_y, val_y = split_data(data, labels, group)
        train_x, test_x, val_x = data_normalize(train_x, test_x, val_x, data)

        # 定义 EarlyStopping 回调
        early_stopping = EarlyStopping(
            monitor='val_r2_score',    
            mode = 'max',
            patience=15,         
            restore_best_weights=True 
        )
        
        model = build_model()
        print('Training....')
        history = model.fit(train_x,
                    train_y, epochs=100, batch_size=16,
                    validation_data=(val_x,val_y),
                    callbacks=[early_stopping],
                    verbose=2
                    )
        
        _,val_r2,val_mae = model.evaluate(val_x,val_y,verbose=0)
        _,test_r2,test_mae = model.evaluate(test_x,test_y,verbose=0)
        
        print('/n/nTraining Done!')
        print('***************** Results of the best model *****************')
        print('val_r2:',val_r2,'val_mae:',val_mae)
        print('test_r2:',test_r2,'test_mae:',test_mae)
        print('*************************************************************')

        # save model
        model.save(f'./model/{target_label}_origin.keras')
        
        # save result
        pre_val = model.predict(val_x)
        val_result = pd.DataFrame({'id':cod_id[group=='val'],'label':val_y,'pred':pre_val.reshape(-1)})
        val_result['data'] = [x for x in val_x]
        val_result['mae'] = np.abs(val_result['label'] - val_result['pred'])

        pre_train = model.predict(train_x)
        train_result = pd.DataFrame({'id':cod_id[group=='train'],'label':train_y,'pred':pre_train.reshape(-1)})
        train_result['data'] = [x for x in train_x]
        train_result['mae'] = np.abs(train_result['label'] - train_result['pred'])

        pre_test = model.predict(test_x)
        test_result = pd.DataFrame({'id':cod_id[group=='test'],'label':test_y,'pred':pre_test.reshape(-1)})
        test_result['data'] = [x for x in test_x]
        test_result['mae'] = np.abs(test_result['label'] - test_result['pred'])

        train_result.to_csv(f'./result/train_result_{target_label}_origin.csv',index=False)
        test_result.to_csv(f'./result/test_result_{target_label}_origin.csv',index=False)
        val_result.to_csv(f'./result/val_result_{target_label}_origin.csv',index=False)

if __name__=='__main__':
    main()
    
    
    
    

