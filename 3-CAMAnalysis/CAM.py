#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   CAM.py
@Time    :   2024/06/14 14:17:34
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""


from skimage import transform
import tensorflow as tf
import numpy as np


def get_heatmap(index,dataset,last_conv_layer_model,pre_model):
    experience_sample = dataset[index]

    img_array = experience_sample.reshape(-1, 64, 64, 3)

    # Compute activations of the last conv layer and make the tape watch it.
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Retrieve the output corresponding to the regression task
        preds = pre_model(last_conv_layer_output)
        # Directly use the predicted value
        top_class_channel = preds[:, 0]

    # This is the gradient of the predicted value with regard to the output feature map of the last convolutional layer.
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # post_preprocessing
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.flip(heatmap, axis=0)

    pic = transform.resize(heatmap, (64, 64), order=2)

    return pic


def get_heatmap_single_channel(index, channel,dataset,last_conv_layer_model,pre_model):
    experience_sample = dataset[index]

    # Create a copy of the input with only one channel active
    img_array = np.zeros_like(experience_sample)
    img_array[:, :, channel] = experience_sample[:, :, channel]
    img_array = img_array.reshape(-1, 64, 64, 3)

    # Compute activations of the last conv layer and make the tape watch it.
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Retrieve the output corresponding to the regression task
        preds = pre_model(last_conv_layer_output)
        # Since it's a regression task, we don't have top_pred_index
        # Directly use the predicted value
        top_class_channel = preds[:, 0]

    # This is the gradient of the predicted value with regard to the output feature map of the last convolutional layer.
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # Post-processing
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.flip(heatmap, axis=0)
    
    
    # heatmap 归一化到[0,1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # print(heatmap.max(), heatmap.min())

    
    pic = transform.resize(heatmap, (64, 64), order=4)
    
    # print(heatmap.shape)
    # pic = heatmap
    return pic

