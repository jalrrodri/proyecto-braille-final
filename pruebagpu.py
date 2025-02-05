import tensorflow as tf
import time

# Define matrix size
matrix_size = 8192

# Create random matrices
a = tf.random.normal((matrix_size, matrix_size))
b = tf.random.normal((matrix_size, matrix_size))

# Warm-up
for _ in range(5):
    _ = tf.matmul(a, b)

# Measure time for matrix multiplication
start = time.time()
_ = tf.matmul(a, b)
end = time.time()

print(f"Matrix multiplication took {end - start:.5f} seconds.")
