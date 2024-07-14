import os
import tensorflow as tf
import numpy as np

# Set environment variables (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['OMP_NUM_THREADS'] = 'your_number_of_threads'  # Adjust to the number of CPU threads you have

# List all available CPUs
physical_cpus = tf.config.experimental.list_physical_devices('CPU')
print(f"Number of CPUs available: {len(physical_cpus)}")

# Ensure TensorFlow is using the CPUs
logical_cpus = tf.config.experimental.list_logical_devices('CPU')
print(f"Number of logical CPUs: {len(logical_cpus)}")

# Confirm CPU usage by running a simple computation
with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(c)

# Configure MultiWorkerMirroredStrategy for CPU training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Define a simple model to test multi-CPU usage
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(896, 896, 18)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Create a dummy data generator to test the setup
class DummyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.x = np.random.rand(batch_size, 896, 896, 18).astype(np.float32)
        self.y = np.random.rand(batch_size, 896, 896, 1).astype(np.float32)

    def __len__(self):
        return 1  # Only one batch for testing

    def __getitem__(self, index):
        return self.x, self.y

# Create a dummy data generator instance
dummy_data_generator = DummyDataGenerator(batch_size=1)

# Fit the model using the dummy data generator
model.fit(dummy_data_generator, epochs=1)