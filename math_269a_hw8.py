import numpy as np
import matplotlib.pyplot as plt

def deriv(t, y):
    dy1dt = 20 / (np.pi * (1 + (20 * y[1] - 100)**2))
    dy2dt = 1
    return np.array([dy1dt, dy2dt])

def huen_step(t, y, h, deriv):
    k1 = h * deriv(t, y)
    k2 = h * deriv(t + h, y + k1)
    return y + 0.5 * (k1 + k2)

def huen_method(deriv, y0, t0, T, h_step):
    # Initialize variables
    t = t0
    y = np.array(y0)
    h = h_step

    # Initialize return arrays
    t_seq = [t0]
    y_step_seq = [y0]
    y_double_seq = [y0]
    err_seq = [0]

    # Main loop
    for i in range(100):
        # Compute two steps with step size h
        y1 = huen_step(t, y, h, deriv)
        y2 = huen_step(t + h, y1, h, deriv)

        # Compute one step with step size 2h
        y_double = huen_step(t, y, 2 * h, deriv)

        # Compute the local error estimate
        error = np.max(np.abs(y_double - y2))

        t += h

        t_seq.append(t)
        y_step_seq.append(y2)
        y_double_seq.append(y_double)
        err_seq.append(error)

    return np.column_stack((t_seq, y_step_seq, y_double_seq, err_seq))

# Example usage
t0 = 0
T = 10
y0 = [(np.arctan(-100) / np.pi) + 0.5, 0]
h_initial = 0.1

result = huen_method(deriv, y0, t0, T, h_initial)

# Display the results
print(result)
