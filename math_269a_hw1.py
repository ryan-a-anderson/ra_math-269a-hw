import numpy as np
import matplotlib.pyplot as plt

def f_deriv(y, t):
    return -1.0 * y + 2 * np.cos(t) + 2 * np.sin(t)

def forward_euler(h_step, y_0, t_0, t_n):
    y_seq = []
    y_seq.append(y_0)
    t_seq = []
    t_seq.append(t_0)
    
    i = 1
    while t_seq[-1] < t_n:
        y_seq.append(y_seq[-1] + h_step * f_deriv(y_seq[-1], t_seq[-1]))
        t_seq.append(t_seq[-1] + h_step)
        i += 1
    
    return np.column_stack((t_seq, y_seq))

# Example usage:
h_step = 0.1
y_0 = 0.0
t_0 = 0.0
t_n = 5.0

result = forward_euler(h_step, y_0, t_0, t_n)

plt.plot(result[:, 0], result[:, 1], label='y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Forward Euler Method')
plt.legend()
plt.grid(True)
plt.show()
