import numpy as np
import matplotlib.pyplot as plt

def f_deriv(t, y):
    return np.array([1, np.cos(t), -y + 2 * np.cos(t) + 2 * np.sin(t)])

def vec_forward_euler(h_step, y_0, t_0, t_n):
    num_steps = int(t_n / h_step)
    vec_seq = np.empty((num_steps, 3))
    vec_seq[0, :] = [t_0, np.cos(t_0), y_0]
    t_seq = np.empty(num_steps)
    t_seq[0] = t_0

    for i in range(1, num_steps):
        vec_seq[i, :] = vec_seq[i - 1, :] + h_step * f_deriv(t_seq[i - 1], vec_seq[i - 1, 2])
        t_seq[i] = t_seq[i - 1] + h_step

    return vec_seq

def vec_average_method(h_step, y_0, t_0, t_n):
    num_steps = int(t_n / h_step)
    vec_seq = np.empty((num_steps, 3))
    vec_seq[0, :] = [t_0, np.cos(t_0), y_0]
    t_seq = np.empty(num_steps)
    t_seq[0] = t_0

    for i in range(1, num_steps):
        vec_star = vec_seq[i - 1, :] + 0.5 * h_step * f_deriv(t_seq[i - 1], vec_seq[i - 1, 2])
        vec_seq[i, :] = vec_seq[i - 1, :] + h_step * f_deriv(t_seq[i - 1], vec_star[2])
        t_seq[i] = t_seq[i - 1] + h_step

    return vec_seq

def g_deriv(t, y):
    return np.array([1, t**2, 4 * t**2 * np.cos(y)])

def other_vec_forward_euler(h_step, y_0, t_0, t_n):
    num_steps = int(t_n / h_step)
    vec_seq = np.empty((num_steps, 3))
    vec_seq[0, :] = [t_0, np.cos(t_0), y_0]
    t_seq = np.empty(num_steps)
    t_seq[0] = t_0

    for i in range(1, num_steps):
        vec_seq[i, :] = vec_seq[i - 1, :] + h_step * g_deriv(t_seq[i - 1], vec_seq[i - 1, 2])
        t_seq[i] = t_seq[i - 1] + h_step

    return vec_seq

def other_vec_average_method(h_step, y_0, t_0, t_n):
    num_steps = int(t_n / h_step)
    vec_seq = np.empty((num_steps, 3))
    vec_seq[0, :] = [t_0, np.cos(t_0), y_0]
    t_seq = np.empty(num_steps)
    t_seq[0] = t_0

    for i in range(1, num_steps):
        vec_star = vec_seq[i - 1, :] + 0.5 * h_step * g_deriv(t_seq[i - 1], vec_seq[i - 1, 2])
        vec_seq[i, :] = vec_seq[i - 1, :] + h_step * g_deriv(t_seq[i - 1], vec_star[2])
        t_seq[i] = t_seq[i - 1] + h_step

    return vec_seq

# Example usage:
result_1 = other_vec_forward_euler(1.0, 1.0, 0.0, 20.0)
result_0_5 = other_vec_forward_euler(0.5, 1.0, 0.0, 20.0)
result_0_25 = other_vec_forward_euler(0.25, 1.0, 0.0, 20.0)
result_0_001 = other_vec_forward_euler(0.001, 1.0, 0.0, 20.0)

t_values = np.arange(1, 20/0.001 + 1, 1)
plt.plot(t_values, result_1[:, 2], 'r', label='h = 1.0')
plt.plot(t_values, result_0_5[:, 2], 'b', label='h = 0.5')
plt.plot(t_values, result_0_25[:, 2], 'g', label='h = 0.25')
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Other Vec Forward Euler Method')
plt.grid(True)
plt.show()

t_values = np.arange(1, 20/0.001 + 1, 1)
result_0_001_2 = other_vec_average_method(0.001, 1.0, 0.0, 20.0)
plt.plot(t_values, result_0_001_2[:, 2], 'b')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Other Vec Average Method')
plt.grid(True)
plt.show()
