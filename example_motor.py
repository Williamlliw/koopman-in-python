import numpy as np
import koopman_module as koopman
from matplotlib import  pyplot as plt
import motor_sys as sys

plt.close('all')

# initialize parameters
nx = 2  # dimension of states
ny = 1  # dimension of outputs
nu = 1  # dimension of control inputs
nd = 1  # delayed number
t_step = 0.01 # time step for simulation
min = -1
max = 1
para_u = [min, max]
para_x = [min, max]
n_sim = 1000
n_traj = 200
n_rbf = 30 # number of RBFs
rbf_type = 'thinplate'


# Basis functions
np.random.seed(42)  # For reproducibility

# define objects
# define koopman object
kp_obj = koopman.KoopmanModel(nx, ny, nu, n_rbf, n_sim, n_traj, rbf_type, n0 = 1, nd = nd)
# define motor object
motor = sys.motor(t_step, kp_obj.n_lift)
f_ud = motor.known_f_ud  # known dynamics function


# Generate data
X, Y, U = kp_obj.generate_data(f_ud, para_x, para_u, motor.C)

# Build lifted predictor
A_lift, B_lift, C_lift, WVT2, VVT2 = kp_obj.regression(X, Y, U)

# Predictor comparison
T_max = 3
n_sim2 = int(T_max / t_step)

# Initial condition
x0 = np.array([[0.5], [0.5]])  # Initial state
x_true = x0
u_rec = np.zeros(n_sim2)  # Record control inputs for EDMD
cost = np.zeros(n_sim2)  # Record cost for EDMD

# Lifted initial condition
delay_y = motor.C @ x0  # Initial delayed state
for i in range(nd):
    urand = np.random.rand()-0.5  # Random control input for delay state
    x_true = motor.known_f_ud(x_true, urand)
    y_true = motor.C @ x_true
    delay_y, flag = kp_obj.delay_state(delay_y, y_true, urand)


# Simulate
x_koop_rec = kp_obj.lift_fun(delay_y)  # Initial state for EDMD


for i in range(n_sim2):
    # lift the delayed state
    X_lift = kp_obj.lift_fun(delay_y)  # Lifted state for EDMD
    # mpc control, solve linear MPC in the lifted space
    u = np.sign(np.random.rand()-0.5)  # random input

    # True dynamics
    x_true = np.hstack([x_true, f_ud(x_true[:, -1:], u)])
    y_true = motor.C @ x_true[:, -1:]
    # koopman dynamics prediction
    x_koop_new = A_lift @ x_koop_rec[:, -1:] + B_lift @ np.array([[u]])
    x_koop_rec = np.hstack([x_koop_rec, x_koop_new])


    # update delay state
    delay_y_old = delay_y
    delay_y, flag = kp_obj.delay_state(delay_y, y_true, u)


    # record control inputs
    u_rec[i] = u
    
x_koop = C_lift @ x_koop_rec



# Plot results
plt.figure('y')
itera_num = np.arange(n_sim2 + 1)
plt.plot(itera_num * t_step, x_true[1, :], label='True y')
plt.plot(itera_num * t_step, x_koop[0, :], label='Koopman y')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.legend()
plt.title('True Dynamics vs Koopman Prediction')
plt.grid()


plt.figure('input')
plt.plot(np.arange(n_sim2) * t_step, u_rec, label='EDMD')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.title('Control Input')
plt.grid()


plt.show()