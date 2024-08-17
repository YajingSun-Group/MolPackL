#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   example.py
@Time    :   2024/06/14 14:21:50
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #use GPU:0 only
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf 
from utils import load_data, data_preprocess, split_data, data_normalize
from CAM import get_heatmap,get_heatmap_single_channel

gpus = tf.config.experimental.list_physical_devices(device_type='GPU') 
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True) 

print('Tensorflow Version:',tf.__version__)


# load data
data0, data1, data2, labels, group,cod_ids = load_data()
data = data_preprocess(data0, data1, data2)
train_x, test_x, val_x, train_y, test_y, val_y = split_data(data, labels, group)
train_x, test_x, val_x = data_normalize(train_x, test_x, val_x, data)


# load model
model  = tf.keras.models.load_model('../1-BandGapPrediction/model_band_gap-final-2.keras')

# get last conv layer 
last_conv_layer_name = "max_pooling2d_14"
predictor_layer_names = [
    "flatten_4",
    "dropout_4",
    "dense_16",
    "dense_17",
    "dense_18",
    "dense_19"]


last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = tf.keras.Model(model.input, last_conv_layer.output)

predictor_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = predictor_input
for layer_name in predictor_layer_names:
    x = model.get_layer(layer_name)(x)
pre_model = tf.keras.Model(predictor_input, x)


# plot  CAM heatmap of test_dataset
for index in range(len(test_x)):
    
    cod_num = cod_ids[group=='test'][index]
    print('num: ',index, 'cod_id: ',cod_num)
    save_path = f'./pic/test/{cod_num}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num = index
    dataset = test_x
    cmap='GnBu'

    # Plot raw img and heatmap for channel 0
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(121)
    plt.imshow(dataset[num][:, :, 0], origin='lower', cmap=cmap)
    plt.title('Raw data')
    plt.axis('off')
    plt.subplot(122)
    heatmap_0 = get_heatmap_single_channel(index=num, channel=0,dataset=dataset,
                                            last_conv_layer_model=last_conv_layer_model,
                                           pre_model=pre_model
                                           )
    plt.imshow(heatmap_0, cmap='jet', vmin=0, vmax=1)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Adjust color bar size
    cbar.ax.tick_params(labelsize=10)  # Adjust color bar tick size
    plt.title('CAM heatmap')
    plt.axis('off')
    plt.savefig(f'{save_path}/channel_0.png')
    plt.close()
    # plt.show()

    # Plot raw img and heatmap for channel 1
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(121)
    plt.imshow(dataset[num][:, :, 1], origin='lower', cmap=cmap)
    plt.title('Raw data')
    plt.axis('off')
    plt.subplot(122)
    heatmap_1 = get_heatmap_single_channel(index=num, channel=1,dataset=dataset,
                                           last_conv_layer_model=last_conv_layer_model,
                                           pre_model=pre_model)
    
    plt.imshow(heatmap_1, cmap='jet', vmin=0, vmax=1)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Adjust color bar size
    cbar.ax.tick_params(labelsize=10)  # Adjust color bar tick size
    plt.title('CAM heatmap')
    plt.axis('off')
    plt.savefig(f'{save_path}/channel_1.png')
    plt.close()


    # Plot raw img and heatmap for channel 2
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(121)
    plt.imshow(dataset[num][:, :, 2], origin='lower', cmap=cmap)
    plt.title('Raw data')
    plt.axis('off')
    plt.subplot(122)
    heatmap_2 = get_heatmap_single_channel(index=num, channel=2,dataset=dataset,
                                        last_conv_layer_model=last_conv_layer_model,
                                           pre_model=pre_model)
    plt.imshow(heatmap_2, cmap='jet', vmin=0, vmax=1)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Adjust color bar size
    cbar.ax.tick_params(labelsize=10)  # Adjust color bar tick size
    plt.title('CAM heatmap')
    plt.axis('off')
    plt.savefig(f'{save_path}/channel_2.png')
    plt.close()


    # Plot raw img and heatmap for channel 2
    plt.figure(figsize=(10, 10), dpi=100)
    plt.subplot(121)
    plt.imshow(dataset[num], origin='lower', cmap=cmap)
    plt.title('Raw data')
    plt.axis('off')
    plt.subplot(122)
    heatmap_3 = get_heatmap(index=num,dataset=dataset,
                            last_conv_layer_model=last_conv_layer_model,
                            pre_model=pre_model)
    plt.imshow(heatmap_3, cmap='jet', vmin=0, vmax=1)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)  # Adjust color bar size
    cbar.ax.tick_params(labelsize=10)  # Adjust color bar tick size
    plt.title('CAM heatmap')
    plt.savefig(f'{save_path}/channel_all.png')
    plt.close()
    

    for i in range(3):
        channel = i

        select_x = dataset[num][:, :, channel]

        plt.figure(figsize=(64,64),dpi=1)
        print('Raw data')
        # plt.title('Raw data of 2DUV')
        plt.axis('off')
        plt.imshow(dataset[num][:, :, channel], origin='lower', cmap=cmap)
        plt.savefig(f'{save_path}/raw_example_channel_{channel}.png')


        # # CAM pic
        plt.figure(figsize=(64,64),dpi=1)
        print('Class Activation Map')
        plt.axis('off')
        heatmap = get_heatmap_single_channel(index=num, channel=channel,dataset=dataset,
                                             last_conv_layer_model=last_conv_layer_model,
                                             pre_model=pre_model)
        plt.imshow(heatmap,cmap = 'jet')
        plt.savefig(f'{save_path}/cam_example_channel_{channel}.png')
        plt.close()
    
        
        img_path = f'{save_path}/raw_example_channel_{channel}.png'
        img = tf.keras.preprocessing.image.load_img(img_path,target_size=(64,64))
        img = tf.keras.preprocessing.image.img_to_array(img)

        hemp_path = f'{save_path}/cam_example_channel_{channel}.png'
        heatmap = tf.keras.preprocessing.image.load_img(hemp_path,target_size=(64,64))
        heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)

        superimposed_img = heatmap * 0.4 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img.save(f'{save_path}/superimposed_img_channel_{channel}.png')

    
    
    # select_x = dataset[num][:, :, channel]

    plt.figure(figsize=(64,64),dpi=1)
    print('Raw data')
    # plt.title('Raw data of 2DUV')
    plt.axis('off')
    plt.imshow(dataset[num], origin='lower', cmap=cmap)
    plt.savefig(f'{save_path}/raw_example_channel_all.png')
    plt.close()

    # # CAM pic
    plt.figure(figsize=(64,64),dpi=1)
    print('Class Activation Map')
    # plt.title('Class Activation Map')
    plt.axis('off')
    heatmap = get_heatmap(index=num,dataset=dataset,
                        last_conv_layer_model=last_conv_layer_model,
                        pre_model=pre_model
                          )
    plt.imshow(heatmap,cmap = 'jet')
    plt.savefig(f'{save_path}/cam_example_channel_all.png')
    plt.close()
    
    img_path = f'{save_path}/raw_example_channel_all.png'
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(64,64))
    img = tf.keras.preprocessing.image.img_to_array(img)


    hemp_path = f'{save_path}/cam_example_channel_all.png'
    heatmap = tf.keras.preprocessing.image.load_img(hemp_path,target_size=(64,64))
    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(f'{save_path}/superimposed_img_channel_all.png')
    
    # save_all data
    np.save(f'{save_path}/data.npy',dataset[num])
    # save heatmap
    np.save(f'{save_path}/heatmap_0.npy',heatmap_0)
    np.save(f'{save_path}/heatmap_1.npy',heatmap_1)
    np.save(f'{save_path}/heatmap_2.npy',heatmap_2)
    np.save(f'{save_path}/heatmap_all.npy',heatmap_3)




    