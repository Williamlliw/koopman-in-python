'''
version 1.0
Author: ycw
Date: 2025-09-08
Description: This module implements the Koopman operator theory for system
            identification and prediction.
'''

import numpy as np
import time
from scipy.linalg import pinv


class KoopmanModel:
    def __init__(self, nx, ny, nu, n_rbf, n_sim, n_traj, rbf_type, n0 = 100, nd = 0):
        '''
        nx: dimension of states
        ny: dimension of outputs
        nu: dimension of control inputs
        n_rbf: number of RBFs
        n_sim: number of time steps for each trajectory
        n_traj: number of trajectories
        rbf_type: type of RBFs, options: 'thinplate', 'gauss', 'invquad', 'invmultquad', 'polyharmonic'
        n0: number of effective data points we attribute to the prior.
        nd: delayed number, if nd > 0, delay state will be generated
        '''
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.nd = nd
        self.delay_dim = (nd + 1) * ny + nu * nd
        self.n_rbf = n_rbf
        self.n_lift = n_rbf + self.delay_dim
        self.n_sim = n_sim
        self.n_traj = n_traj
        self.n0 = n0
        
        np.random.seed(42)  # For reproducibility
        self.cent = np.random.rand(self.delay_dim, n_rbf) * 2 - 1 # 生成[-1, 1]的随机数
        self.rbf_type = rbf_type



    # observables function
    def lift_fun(self, xx):
        lift_x = np.vstack([xx, self.rbf(xx, self.cent, self.rbf_type)])
        return lift_x
        

    def predict(self, x, u):
        z = self.lift_fun(x)
        z_next = self.A_lift @ z + self.B_lift @ u
        x_next = self.C_lift @ z_next
        return x_next


    def rbf(self, X, C, type, eps=None, k=None):
        '''
        n: dimension of state
        N: size of data set
        X: nxN
        C: rbf center(s) - nx1 or nxK, if C is n x K, then the result is STACKED
        over the centers K, i.e., of dimension K x N
        eps: kernel width for Gaussian type rbfs (optional)
        k: polyharmonic coefficient for polyharmonic rbfs (optional)
        Note that all RBFs are symmetrical so evaluation of of a single point on
        multiple centers can by done as evaluation of multiple points (the
        centers in this case) on a single point (the center)
        '''

        type = type.lower()
        if eps is None or eps == []:
            eps = 1
        if k is None or k == []:
            k = 1

        Cbig = C
        Y = np.zeros((Cbig.shape[1], X.shape[1]))
        for i in range(Cbig.shape[1]):
            C = Cbig[:, i:i+1]  # Extract the i-th column as a 2D array
            C = np.tile(C, (1, X.shape[1]))  # Repeat the column to match X's shape
            r_squared = np.sum((X - C) ** 2, axis=0)
            if type == 'thinplate':
                y = r_squared * np.log(np.sqrt(r_squared))
                y[np.isnan(y)] = 0
            elif type == 'gauss':
                y = np.exp(-eps**2 * r_squared)
            elif type == 'invquad':
               y = 1 / (1 + eps**2 * r_squared)
            elif type == 'invmultquad':
                y = 1 / np.sqrt(1 + eps**2 * r_squared)
            elif type == 'polyharmonic':
                y = r_squared ** (k / 2) * np.log(np.sqrt(r_squared))
                y[np.isnan(y)] = 0
            else:
                raise ValueError('RBF type not recognized')
            Y[i, :] = y
        return Y


    # given dynamic function, generate data sets along several trajectories
    def generate_data(self, f_ud, para_x, para_u, C, initial_state=None):
        '''
        input:
        f_ud: function handle, f_ud(x,u) returns the next state given current
        state and control input
        n_sim: number of time steps
        n_traj: number of trajectories
        para_x: a list of [n, min_x, max_x],
            where: n: dimension of states, min_x, max_x: range of states
        para_u: a list of [m, min_u, max_u],
            where: m: dimension of control, min_u, max_u: range of control
        C: out put matrix for dynamic function
        initial_state: initial state for the first trajectory, if None, random
        initial state will be generated
        nd: delayed number, if nd > 0, delay state will be generated

        output:
        X: n x (n_sim*n_traj), state data set
        Y: n x (n_sim*n_traj), next state data set
        U: m x (n_sim*n_traj), control input data set
        '''

        # part 1: initialization
        # get dimension and min max of states and control inputs
        n_sim = self.n_sim
        n_traj = self.n_traj
        nd = self.nd
        n = self.nx
        m = self.nu
        min_x, max_x = para_x
        min_u, max_u = para_u
        ny = C.shape[0]  # dimension of output

        # Collect data
        start_time = time.time()
        print('Generating data')

        # initialize U, X, Y
        # random input, range in [min_u, max_u]
        U = (max_u - min_u) * np.random.rand(m, n_sim*n_traj) + min_u
        # random initial state, range in [min_x, max_x]
        if initial_state is not None:
            if initial_state.shape[0] != n or initial_state.shape[1] != n_traj:
                raise ValueError("Initial state shape must be (n, n_traj)")
            initial_X = initial_state
        else:
            # Random initial conditions, range in [min_x, max_x]
            initial_X = (max_x - min_x) * np.random.rand(n, n_traj) + min_x
    

        # part 2: generate data along the trajectory
        if nd == 0:
            # part 2.1: No delay state
            # Initialize X and Y, which will store the generated data
            # X_current, X_next is the current and next state respectively
            X = initial_X
            Y = f_ud(initial_X, U[:, :n_traj])
            X_current = Y
            for ii in range(n_sim-1):
                # dynamic, use X_current and U to get X_next
                X_next = f_ud(X_current, U[:, (ii+1)*n_traj:(ii+2)*n_traj])
                X = np.hstack((X, X_current))
                Y = np.hstack((Y, X_next))
                X_current = X_next

                # Check for NaN and inf values
                if np.isnan(X_next).any():
                    raise ValueError("输入值中包含 NaN")
                if np.isinf(X_next).any():
                    raise ValueError("输入值中包含 inf")
        else:
            # part 2.2: With delay state
            delay_dim = self.delay_dim
            X = np.zeros((delay_dim, 0))  # Initialize X for storing delayed states
            Y = np.zeros((delay_dim, 0))  # Initialize Y for storing next states
            X_current = initial_X  # Current state
            delay_current = C @ initial_X
            for ii in range(n_sim):
                X_next = f_ud(X_current, U[:, ii*n_traj:(ii+1)*n_traj])
                y_next = C @ X_next  # Get next state output
                # Delay state
                delay_next, flag = self.delay_state(delay_current, y_next, U[:, ii*n_traj:(ii+1)*n_traj])
                if flag:
                    # If the delayed state is full, append to X and Y
                    X = np.hstack((X, delay_current))
                    Y = np.hstack((Y, delay_next))
                # Update current delay state
                delay_current = delay_next
                X_current = X_next

                # Check for NaN and inf values
                if np.isnan(X_next).any():
                    raise ValueError("输入值中包含 NaN")
                if np.isinf(X_next).any():
                    raise ValueError("输入值中包含 inf")
    
        print(f'Data generation DONE, time = {time.time() - start_time:.2f} s')
        return X, Y, U[:, -X.shape[1]:]  # Return U with the same number of columns as X


    def delay_state(self, delay_y, next_y, current_u):
        """
        this function computes a delay-embedded state for a system.

        input:
        nd: delayed number
        delay_y: delayed state of previous step
        para: [ny, nu], dimension of output and input respectively
        output:
        delay_y: current delayed state
        flag: true if delayed state is full (initialization completed), false otherwise

        format of delay-embedded "state" :
        zeta_k = [y_{k} ; u_{k-1} ; y_{k-1} ... u_{k-nd} ; y_{k-nd} ];
        """
        ny = self.ny
        nu = self.nu
        nd = self.nd

        # flag=true if delayed state is full (initialization completed), false otherwise
        if delay_y.shape[0] <= (ny * nd + (nd-1) * nu):
            flag = False
            delay_y = np.vstack((next_y, current_u, delay_y))
        else:
            flag = True
            delay_y = np.vstack((next_y, current_u, delay_y[0: -ny-nu, :]))

        return delay_y, flag


    # regression to build lifted predictor
    def regression(self, X, Y, U, lift_fun=None):
        '''
        Build lifted predictor
        input:
        lift_fun: function handle, lift_fun(x) returns lifted state
        X: n x (n_sim*n_traj), state data set
        Y: n x (n_sim*n_traj), next state data set
        U: m x (n_sim*n_traj), control input data set
        n_lift: number of lifted states
        output:
        A_lift: lifted A matrix
        B_lift: lifted B matrix
        C_lift: lifted C matrix
        predictor format:
        dot_z = A_lift * z + B_lift * u
        x = C_lift * z
        '''

        # part 1: get private parameters
        n_lift = self.n_lift

        if lift_fun is None:
            lift_fun = self.lift_fun

        start_time = time.time()
        print('Starting LIFTING')
        # Lift the data
        X_lift = lift_fun(X)
        Y_lift = lift_fun(Y)
        print(f'Lifting DONE, time = {time.time() - start_time:.2f} s')

        # Build predictor
        start_time = time.time()
        print('Starting REGRESSION')    
        W = np.vstack([Y_lift, X])
        V = np.vstack([X_lift, U])
        VVt = V @ V.T
        WVt = W @ V.T
        M = WVt @ pinv(VVt)  # Matrix [A B; C 0]
        A_lift = M[:n_lift, :n_lift]
        B_lift = M[:n_lift, n_lift:]
        C_lift = M[n_lift:, :n_lift]

        print(f'Regression DONE, time = {time.time() - start_time:.2f} s')

        res_norm = np.linalg.norm(Y_lift - A_lift @ X_lift - B_lift @ U, ord='fro') \
            / np.linalg.norm(Y_lift, ord='fro')
        print(f"Regression residual = {res_norm:.6f}")

        return A_lift, B_lift, C_lift, WVt, VVt

def discretize_system(f, x_k, u_k, h):
    """
    使用四阶龙格-库塔法，将连续的动力学方程离散化
    
    参数:
    f: 函数，表示连续系统的微分方程 dx/dt = f(x, u)
    x_k: 当前时间步的状态变量 x_k
    u_k: 当前时间步的输入变量 u_k
    h: 时间步长
    
    返回:
    x_{k+1}: 下一个时间步的状态变量
    """
    k1 = f(x_k, u_k)
    k2 = f(x_k + 0.5*h*k1, u_k)
    k3 = f(x_k + 0.5*h*k2, u_k)
    k4 = f(x_k + k3*h, u_k)
    
    x_k1 = x_k + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # 假设 x1, x2, u 是 NumPy 数组
    if np.isnan(x_k1).any() or np.isnan(x_k).any() or np.isnan(u_k).any():
        raise ValueError("there's NaN in the input values")

    if np.isinf(x_k1).any() or np.isinf(x_k).any() or np.isinf(u_k).any():
        raise ValueError("there's inf in the input values")
    return x_k1