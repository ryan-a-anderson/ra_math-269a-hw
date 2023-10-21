import numpy as np
import matplotlib.pyplot as plt

def f_deriv(t, y):
    return np.array([1, 2*np.cos(t), -y + 2 * np.cos(2*t) + 2 * np.sin(2*t)])

def aut_func(t):
    return 2*np.cos(2*t)

def fourth_runge_kutta(h_step,y_0,t_0,t_n):
    num_steps = int(t_n / h_step)
    vec_seq = np.empty((num_steps+1, 3))
    v_t = aut_func(t_0)
    vec_seq[0, :] = [t_0, v_t, y_0]
    t_seq = np.empty(num_steps+1)
    t_seq[0] = t_0

    for i in range(1, num_steps+1):
        y_old = vec_seq[i-1,:]
        k1 = f_deriv(y_old[0],y_old[2])
        k2 = f_deriv(y_old[0],y_old[2]+0.5*h_step*k1[2])
        k3 = f_deriv(y_old[0],y_old[2]+0.5*h_step*k2[2])
        k4 = f_deriv(y_old[0],y_old[2]+0.5*h_step*k3[2])

        y_new = y_old[2] + (h_step/6)*(k1[2]+k4[2]) + (h_step/3)*(k2[2]+k3[2])
        vec_seq[i,:] = [vec_seq[i-1,0]+h_step,aut_func(vec_seq[i-1,0]+h_step),y_new]
    
    return vec_seq

print(fourth_runge_kutta(0.0001,1.0,0.0,8.0))