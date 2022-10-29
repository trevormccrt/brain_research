import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def ode_fn(_, y, W):
    return -y + tf.einsum("ij, ...j -> ...i", W, tf.sigmoid(y))

batch_size = 5
n = 500
init_states = np.random.uniform(-0.1, 0.1, (batch_size, n))
weights = tf.Variable(np.random.uniform(-1, 1, int(n * (n+1)/2)))
solver = tf.function(tfp.math.ode.BDF().solve)
eq_time = 5
soln_times = tf.convert_to_tensor([eq_time])
init_time = tf.convert_to_tensor(0)

with tf.GradientTape() as tape:
    tape.watch(weights)
    weight_matrix_upper = tfp.math.fill_triangular(weights)
    weight_matrix = weight_matrix_upper + tf.transpose(weight_matrix_upper)
    results = solver(ode_fn, init_time, init_states, solution_times=soln_times, constants={'W': weight_matrix})
print(results.states)
print("")
