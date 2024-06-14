#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py
@Time    :   2024/06/2 10:22:31
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input


def build_model():
    """
    Build and compile a convolutional neural network model for band gap prediction.

    Returns:
        model (tf.keras.Model): The compiled model for band gap prediction.
    """
    inputs = Input(shape=(64, 64, 3))  # 明确定义输入层
    x = layers.Conv2D(16, (3,3), activation='relu',)(inputs)
    x = layers.MaxPooling2D(pool_size= (2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size= (2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size= (2,2))(x)
    x = layers.Flatten()(x)
    # dropout layer
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.R2Score(),'mae'])
    return model

