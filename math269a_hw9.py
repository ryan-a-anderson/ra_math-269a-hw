import numpy as np
import matplotlib.pyplot as plt

def deriv(y, t):
    y_1 = 20 / (np.pi * (1 + (20*y[1]-100)**2))
    y_2 = 1
    return np.array([y_1, y_2])

def deriv_2(y, t):
    y_1 = 10.0 * (y[0] - y[0]**3/3 - y[1])
    y_2 = 1 / 10.0 * y[0]
    return np.array([y_1, y_2])

t_0 = 0
t_1 = 10
tol = 1e-4
h = 0.5

y_01 = np.array([(1/np.pi)*np.arctan(-100)+0.5, 0])
y_02 = np.array([0.5, 0.5])
t = t_0
y = y_01
err = []

while t < t_1:
    # Steps of size h
    y_1 = y + h * deriv(y, t)
    y_2 = y + h * deriv(y_1, t + h)
    
    # Step of size 2h
    y_3 = y + 2*h * deriv(y, t)
    
    # Error
    temp_err = np.linalg.norm(y_2 - y_3, ord=2)
    err.append(temp_err)
    
    # Check if error is within tolerance
    if temp_err > tol:
        # Reject
        h = h / 2
        continue
    
    # Accept
    if temp_err < tol / 2:
        h = h * 2
    
    t = t +
