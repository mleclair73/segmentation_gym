import os
import time
import platform
import numpy as np

num_epochs = 3
USE_GPU = True
USE_MULTI_GPU = False

# Check for Apple Silicon
is_apple_silicon = platform.system() == 'Darwin' and platform.processor() == 'arm'

if USE_MULTI_GPU:
    GPU_ID = '1,2'
else:
    GPU_ID = '0'

if not is_apple_silicon:
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
else:
    print("Apple Silicon detected, Metal will be used for GPU acceleration")

import tensorflow as tf
import tensorflow.keras as keras

# Only use mixed precision with CUDA
if not is_apple_silicon and USE_GPU:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

# Helper to verify GPU availability
def verify_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        if is_apple_silicon:
            print(f"Metal acceleration available. Found {len(gpus)} GPU device(s):")
        else:
            print(f"CUDA acceleration available. Found {len(gpus)} GPU device(s):")
        print(gpus)
        return True
    else:
        print("No GPU acceleration found. Using CPU only.")
        return False

has_gpu = verify_gpu()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)

if physical_devices:
    try:
        tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
    except RuntimeError as e:
        print(e)

for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

print(tf.config.get_visible_devices())

# Test to verify the GPU is actually being used
def test_gpu_in_use():
    # Create a random matrix operation
    print("Testing if GPU is being used...")
    x = tf.random.normal([5000, 5000])
    
    # Get current device
    print(f"Current device for tensor: {x.device}")
    
    # Force execution to verify placement
    start = time.time()
    result = tf.matmul(x, x)
    # Force execution with a small get operation
    _ = result[0, 0].numpy()
    end = time.time()
    
    print(f"Matrix multiplication took: {end - start:.4f} seconds")
    print(f"Result tensor device: {result.device}")
    
    # Should contain "GPU" or "TPU" if using hardware acceleration
    return "GPU" in result.device or "TPU" in result.device

# Run CPU vs GPU comparison
def run_timing_comparison():
    # Matrix sizes to test
    sizes = [1000, 2000, 4000]
    
    results = {"size": [], "cpu_time": [], "gpu_time": []}
    
    for size in sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        
        # CPU test
        with tf.device('/CPU:0'):
            x_cpu = tf.random.normal([size, size])
            
            # Warmup
            _ = tf.matmul(x_cpu, x_cpu)
            
            # Timed run
            start = time.time()
            result_cpu = tf.matmul(x_cpu, x_cpu)
            _ = result_cpu[0, 0].numpy()  # Force execution
            cpu_time = time.time() - start
            
            print(f"CPU time: {cpu_time:.4f} seconds")
        
        # GPU test
        with tf.device('/GPU:0'):
            x_gpu = tf.random.normal([size, size])
            
            # Warmup
            _ = tf.matmul(x_gpu, x_gpu)
            
            # Timed run
            start = time.time()
            result_gpu = tf.matmul(x_gpu, x_gpu)
            _ = result_gpu[0, 0].numpy()  # Force execution
            gpu_time = time.time() - start
            
            print(f"GPU time: {gpu_time:.4f} seconds")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        results["size"].append(size)
        results["cpu_time"].append(cpu_time)
        results["gpu_time"].append(gpu_time)
    
    return results

# Regular ML model code
def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        run_eagerly=True,
        steps_per_execution=2
    )
    return model

def get_dataset():
    batch_size = 400
    num_val_samples = 2000

    # Return the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

# Run GPU verification
is_using_gpu = test_gpu_in_use()
print(f"Is GPU being used: {is_using_gpu}")

# Run timing comparison 
timing_results = run_timing_comparison()

if USE_MULTI_GPU and not is_apple_silicon:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = get_compiled_model()
else:
    model = get_compiled_model()

# Train the model - with timing to compare
train_dataset, val_dataset, test_dataset = get_dataset()

# Time CPU training
with tf.device('/CPU:0'):
    cpu_model = get_compiled_model()
    start = time.time()
    cpu_model.fit(train_dataset, epochs=1, validation_data=val_dataset)
    cpu_train_time = time.time() - start
    print(f"CPU training time: {cpu_train_time:.2f} seconds")

# Time GPU training
with tf.device('/GPU:0'):
    gpu_model = get_compiled_model()
    start = time.time()
    gpu_model.fit(train_dataset, epochs=1, validation_data=val_dataset)
    gpu_train_time = time.time() - start
    print(f"GPU training time: {gpu_train_time:.2f} seconds")

print(f"Training speedup: {cpu_train_time/gpu_train_time:.2f}x")

# Regular training
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
model.evaluate(test_dataset)