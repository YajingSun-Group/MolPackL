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
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.R2Score(),'mae'])
    return model

# freeze some layers of base model
def freeze_layer(base_model,layer_name = []):
    
    base_model.trainable=True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name in layer_name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    return base_model


def build_fine_tune_model(base_model,hidden_layers_num = [64],drop_out=0.1,lr=5e-5):
    '''
    Pars:
        hidden_layers_num: type:list
        drop_out: rate of drop out layer
        lr: learning rate
    '''
    
    new_model = tf.keras.models.Sequential()
    new_model.add(base_model)
    # new_model.add(tf.keras.layers.Flatten())
    new_model.add(tf.keras.layers.Dropout(drop_out))
    for num_ in hidden_layers_num:
        new_model.add(tf.keras.layers.Dense(num_,activation='relu'))
    new_model.add(tf.keras.layers.Dense(1))
    new_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=lr), metrics=[tf.keras.metrics.R2Score(),'mae'])
    return new_model


