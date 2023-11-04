 import numpy as np
import matplotlib.pyplot as plt

def trapezoidal_method(h_step, y_0, t_0, t_n, deriv):
    num_steps = int(t_n / h_step)
    y_seq = np.zeros(num_steps)
    y_seq[0] = y_0
    t_seq = np.zeros(num_steps)
    t_seq[0] = t_0

    for i in range(1, num_steps):
        y_seq[i] = y_seq[i - 1] + h_step / 2 * (deriv(y_seq[i - 1], t_seq[i - 1]) + deriv(y_seq[i - 1] + h_step * deriv(y_seq[i - 1], t_seq[i - 1]), t_seq[i - 1] + h_step))
        t_seq[i] = t_seq[i - 1] + h_step

    return np.column_stack((t_seq, y_seq))

def curtiss_time_deriv(y, t):
    return -50 * (y - np.cos(t))

def curtiss_exact_solution(t):
    return (2500/2501) * np.cos(t) + (50/2501) * np.sin(t) + (1/2501) * np.exp(-50 * t)

# Perform Trapezoidal Method for different step sizes
trapezoidal_results_h_1_0 = trapezoidal_method(1.0, 1, 0, 10, curtiss_time_deriv)
trapezoidal_results_h_0_1 = trapezoidal_method(0.1, 1, 0, 10, curtiss_time_deriv)
trapezoidal_results_h_0_05 = trapezoidal_method(0.05, 1, 0, 10, curtiss_time_deriv)
trapezoidal_results_h_0_01 = trapezoidal_method(0.01, 1, 0, 10, curtiss_time_deriv)
trapezoidal_results_h_0_001 = trapezoidal_method(0.001, 1, 0, 10, curtiss_time_deriv)

# Plot each result on the same plot with different colors for each time step
plt.figure(figsize=(8, 6))
plt.plot(trapezoidal_results_h_1_0[:, 0], trapezoidal_results_h_1_0[:, 1], 'r', label='h=1.0')
plt.plot(trapezoidal_results_h_0_1[:, 0], trapezoidal_results_h_0_1[:, 1], 'g', label='h=0.1')
plt.plot(trapezoidal_results_h_0_05[:, 0], trapezoidal_results_h_0_05[:, 1], 'purple', label='h=0.05')
plt.plot(trapezoidal_results_h_0_01[:, 0], trapezoidal_results_h_0_01[:, 1], 'b', label='h=0.01')
plt.plot(trapezoidal_results_h_0_001[:, 0], trapezoidal_results_h_0_001[:, 1], 'orange', label='h=0.001')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.ylim(-1, 1)
plt.xlim(0, 10)
plt.title('Trapezoidal Method Results for Curtiss-Hirschfelder Equation')
plt.legend()
plt.show()

# Calculate errors for different step sizes
h_seq = [1.0 / 10**i for i in range(1, 7)]
trapezoidal_error_seq = []

for h in h_seq:
    result = trapezoidal_method(h, 1.0, 0, 10, curtiss_time_deriv)
    error = abs(result[-1, 1] - curtiss_exact_solution(10))
    trapezoidal_error_seq.append(error)
    print("h =", h)

trapezoidal_error_data = np.column_stack((h_seq, trapezoidal_error_seq))
print(trapezoidal_error_data)
