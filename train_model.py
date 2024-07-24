import os
import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import json

# Ensure GPU is used
physical_devices = tf.config.list_physical_devices('GPU')
print('GPUs: ',physical_devices)

physical_devices = tf.config.list_physical_devices('CPU')
print('CPUs: ',physical_devices)

#######################################################################
# -----------------------Define the model------------------------------
#######################################################################

def conv_block(x, n_filters):
    """Two convolutions"""
    x = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(x)
    return x
    
def encoder_block(x, n_filters):
    """Conv block and max pooling"""
    x = conv_block(x, n_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p  # We will need x for the skip connections later

def decoder_block(x, p, n_filters):
    """Upsample, skip connection, and conv block"""
    x = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, p])  # Concatenate = skip connection
    x = conv_block(x, n_filters)
    return x

def unet_model(img_height, img_width, img_channels):
    inputs = Input((img_height, img_width, img_channels))  # 512x512x3
    
    # Contraction path, encoder
    c1, p1 = encoder_block(inputs, n_filters=64)  # c1=512x512x64 p1=256x256x64
    c2, p2 = encoder_block(p1, n_filters=128)  # c2=256x256x128 p2=128x128x128
    c3, p3 = encoder_block(p2, n_filters=256)  # c3=128x128x256 p3=64x64x256
    c4, p4 = encoder_block(p3, n_filters=512)  # c4=64x64x512 p4=32x32x512

    # Bottleneck
    bridge = conv_block(p4, n_filters=1024)  # bridge=32x32x1024
    
    # Expansive path, decoder
    u4 = decoder_block(bridge, c4, n_filters=512)  # 64x64x512
    u3 = decoder_block(u4, c3, n_filters=256)  # 128x128x256
    u2 = decoder_block(u3, c2, n_filters=128)  # 256x256x128
    u1 = decoder_block(u2, c1, n_filters=64)  # 512x512x64

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)  # 512x512x1

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Define Focal Binary Cross-Entropy Loss
def focal_binary_crossentropy(y_true, y_pred, gamma=2., alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.clip_by_value(y_true, epsilon, 1. - epsilon)

    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    focal = focal_binary_crossentropy(y_true, y_pred)
    return bce + focal

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())

# Instantiate and compile the model with an initial learning rate
initial_learning_rate = 0.001
optimizer = Adam(learning_rate=initial_learning_rate)

model_directory = '/home/ayadav7/models/epoch-60/'
model = tf.keras.models.load_model(model_directory, custom_objects={
    'conv_block': conv_block,
    'encoder_block': encoder_block,
    'decoder_block': decoder_block,
    'dice_coefficient': dice_coefficient,
    'combined_loss':combined_loss
})

# model = unet_model(img_height=512, img_width=512, img_channels=16)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient])

print(model.summary())

#######################################################################
# -----------------------Load the data---------------------------------
#######################################################################

def load_data(data_folder):
    X_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('X_')])
    y_files = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('y_')])
    
    X_data = []
    y_data = []
    
    for X_file, y_file in tqdm(zip(X_files, y_files), total=len(X_files)):
        X_data.append(np.load(X_file))
        y_data.append(np.load(y_file))
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    return X_data, y_data



data_folder = '/home/ayadav7/training_data_subset'
X, y = load_data(data_folder)
print(f"Loaded X data shape: {X.shape}")
print(f"Loaded y data shape: {y.shape}")

t1 = datetime.now()
# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print('This took {} seconds'.format(datetime.now() - t1))

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

t1 = datetime.now()
batch_size = 16
train_dataset = create_dataset(X_train, y_train, batch_size)
val_dataset = create_dataset(X_val, y_val, batch_size)
print('This took {} seconds'.format(datetime.now() - t1))

print(f"Training dataset: {train_dataset}")
print(f"Validation dataset: {val_dataset}")

for batch in train_dataset.take(1):
    X_batch, y_batch = batch
    print(f"X_batch shape: {X_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    
#######################################################################
# -----------------------Train the model-------------------------------
#######################################################################

# Define a ModelCheckpoint callback to save the model every 10 epochs
checkpoint_filepath = '/home/ayadav7/models/'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
    initial_value_threshold=None
)

# Define a learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with the learning rate scheduler
history = model.fit(train_dataset, epochs=40, validation_data=val_dataset, callbacks=[reduce_lr, model_checkpoint_callback])

# Save the training history
history_dict = history.history
json.dump(history_dict, open('/home/ayadav7/models/training_history.json', 'w'))

# Save the model
model.save('/home/ayadav7/models/unet_model_final.h5')