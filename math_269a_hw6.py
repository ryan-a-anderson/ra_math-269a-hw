import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, x0, t0, t1, h):
    t_values = np.arange(t0, t1 + h, h)
    x_values = np.zeros((len(t_values), len(x0)))
    x_values[0] = x0

    for i in range(len(t_values) - 1):
        x_values[i + 1] = x_values[i] + h * f(t_values[i], x_values[i])

    return t_values, x_values

def vector_forward_euler(f, x0, t0, t1, h, gamma=1):
    t_values = np.arange(t0, t1 + h, h)
    x_values = np.zeros((len(t_values), len(x0)))
    x_values[0] = x0

    for i in range(len(t_values) - 1):
        x_values[i + 1] = x_values[i] + h * f(t_values[i], x_values[i], gamma)

    return t_values, x_values

def van_der_pol(t, x, gamma=1):
    dv1 = gamma * (x[0] - x[0]**3 / 3 - x[1])
    dv2 = x[0] / gamma
    return np.array([dv1, dv2])

vdp_x0 = np.array([0.5, 0.5])
vdp_t0 = 0
vdp_t1 = 20
h_seq = 2.0**(-np.array(range(8)))

plt.figure(figsize=(16, 4))
for i, h in enumerate(h_seq):
    t, x = vector_forward_euler(van_der_pol, vdp_x0, vdp_t0, vdp_t1, h, gamma=10.0)
    plt.subplot(1, 8, i + 1)
    plt.plot(x[:, 0], x[:, 1], label=f'h={h}')
    plt.xlabel('v_1')
    plt.ylabel('v_2')
    plt.legend()

plt.show()

def ode_two(t, x, gamma):
    du = 1/100 - (1/100 + x[0] + x[1]) * (1 + (x[0] + 1000) * (x[0] + 1))
    dv = 1/100 - (1/100 + x[0] + x[1]) * (1 + x[1]**2)
    return np.array([du, dv])

x0 = np.array([0, 0.5])
t0 = 0
t1 = 5
h_seq = 10.0**(-np.array(range(6)))

plt.figure(figsize=(12, 4))
for i, h in enumerate(h_seq):
    t, x = vector_forward_euler(ode_two, x0, t0, t1, h)
    plt.subplot(1, 6, i + 1)
    plt.plot(x[:, 0], x[:, 1], label=f'h={h}')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.legend()

plt.show()
