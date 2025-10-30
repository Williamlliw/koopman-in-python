import numpy as np
from koopman_module import discretize_system

class motor:
    C = np.zeros((1, 2))  # Output matrix for MPC, to be set later
    C[0, 1] = 1  # Only measure x2

    def __init__(self, t_step, n_lift):
        self.t_step = t_step
        self.n_lift = n_lift

    # Dynamics
    def known_fu(self, x, u):
        # dynamic
        x1 = x[0:1, :]
        x2 = x[1:2, :]
        dot_x = np.zeros((2, x.shape[1]))
        u = u + np.random.randn(x.shape[1])*0.4  # add noise to input
        dot_x[0, :] = 19.10828025-39.3153*x1-32.2293*x2*u
        dot_x[1, :] = -3.333333333-1.6599*x2+22.9478*x1*u
        
        return dot_x
    

    # Discretization using Runge-Kutta method
    def known_f_ud(self, x, u):
        return discretize_system(self.known_fu, x, u, self.t_step)
    