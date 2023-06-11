import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class PINN:
    def __init__(self, layers, optimizer, r, sigma):
        self.model = tf.keras.Sequential(layers)
        self.optimizer = optimizer
        self.r = r  # risk-free interest rate
        self.sigma = sigma  # volatility

    def bsm_residual(self, S, t, V, V_S, V_t, V_SS):
        return V_t + 0.5 * self.sigma**2 * S**2 * V_SS + self.r * S * V_S - self.r * V

    def loss_fn(self, S, t, V):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([S, t])
            V_pred = self.model(tf.stack([S, t], axis=1))
            V_S, V_t = tape.gradient(V_pred, [S, t])
        V_SS = tape.gradient(V_S, S)
        del tape
        residual = self.bsm_residual(S, t, V_pred, V_S, V_t, V_SS)
        return tf.reduce_mean(tf.square(V - V_pred)) + tf.reduce_mean(tf.square(residual))

    def train_step(self, S, t, V):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(S, t, V)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, S, t, V, epochs):
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        V = tf.convert_to_tensor(V, dtype=tf.float32)
        for epoch in range(epochs):
            loss = self.train_step(S, t, V)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    def predict(self, S, t):
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        return self.model(tf.stack([S, t], axis=1))

    def plot(self, S, t, V_true):
        V_pred = self.predict(S, t)
        plt.figure()
        plt.plot(S, V_true, 'r*', label='True')
        plt.plot(S, V_pred, 'b*', label='Predicted')
        plt.legend()
        plt.show()

# Define the neural network architecture and optimizer
layers = [
    tf.keras.layers.Dense(200, activation=tf.nn.tanh, input_shape=(2,)),  # input shape is 2 for (S, t)
    tf.keras.layers.Dense(200, activation=tf.nn.tanh),
    tf.keras.layers.Dense(200, activation=tf.nn.tanh),
    tf.keras.layers.Dense(200, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
]
optimizer = tf.keras.optimizers.Adam()

# Create the PINN
pinn = PINN(layers, optimizer, r=0.05, sigma=0.2)

# Generate some training data (replace this with your actual data)
def generate_data(N):
    # Generate random stock prices, times to maturity, and volatilities
    S = np.random.uniform(50, 150, size=(N, 1))  # stock prices between $50 and $150
    T = np.random.uniform(0.5, 2, size=(N, 1))  # times to maturity between 0.5 and 2 years
    sigma = np.random.uniform(0.1, 0.3, size=(N, 1))  # volatilities between 10% and 30%

    # Use the Black-Scholes-Merton formula to generate option prices
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)  # call option prices

    return S, T, V

# Example usage:
N = 1000  # number of data points
K = 100  # strike price
r = 0.05  # risk-free interest rate
S_train, t_train, V_train = generate_data(N)


# Train the PINN
pinn.train(S_train, t_train, V_train, epochs=1000)

# Plot the results
pinn.plot(S_train, t_train, V_train)
