import numpy as np
from koopman_module import discretize_system

class motor:
    C = np.eye(2)  # measure both states

    def __init__(self, t_step, n_lift):
        self.t_step = t_step
        self.n_lift = n_lift

    # Dynamics
    def known_fu(self, x, u):
        # dynamic
        x1 = x[0:1, :]
        x2 = x[1:2, :]
        dot_x = np.zeros((2, x.shape[1]))
        u = u + np.random.randn(x.shape[1])*0.3
        dot_x[0, :] = 2 * x2
        dot_x[1, :] = -0.8 * x1 - 10 * x1**2 * x2 + 2 * x2 - u
        
        return dot_x
    

    # Discretization using Runge-Kutta method
    def known_f_ud(self, x, u):
        return discretize_system(self.known_fu, x, u, self.t_step)
    
    