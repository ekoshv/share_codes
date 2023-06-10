import tensorflow as tf
import numpy as np

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation=tf.nn.tanh, input_shape=(2,)),  # input shape is 2 for (S, t)
    tf.keras.layers.Dense(20, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Define the known Black-Scholes-Merton (BSM) equation parameters
r = 0.05  # risk-free interest rate
sigma = 0.2  # volatility

# Define the loss function
def loss_fn(S, t, V):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([S, t])
        V_pred = model(tf.stack([S, t], axis=1))
        V_S = tape.gradient(V_pred, S)
        V_t = tape.gradient(V_pred, t)
        V_SS = tape.gradient(V_S, S)
    del tape
    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V_pred
    return tf.reduce_mean(tf.square(V - V_pred)) + tf.reduce_mean(tf.square(residual))

# Define the training step
@tf.function
def train_step(S, t, V):
    with tf.GradientTape() as tape:
        loss = loss_fn(S, t, V)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Generate some training data (replace this with your actual data)
S_train = np.random.rand(1000, 1)
t_train = np.random.rand(1000, 1)
V_train = np.maximum(S_train - 1, 0)  # replace this with your actual option value

# Convert to TensorFlow tensors
S_train = tf.convert_to_tensor(S_train, dtype=tf.float32)
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
V_train = tf.convert_to_tensor(V_train, dtype=tf.float32)

# Train the model
for epoch in range(1000):
    loss = train_step(S_train, t_train, V_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
