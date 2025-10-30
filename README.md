# koopman-in-python
A module for Koopman Operator written in python.

In file "koopman_module.py": basic class and functions of koopman operator;

In file "example_motor.py": run this script to execute an example of koopman operator in motor system;
In file "motor_sys.py": true dynamic of motor system;

In file "example_vdp.py": run this script to execute an example of koopman operator in Van Der Pol system;
In file "vdp_sys.py": true dynamic of Van Der Pol system.

Figure below shows the result of koopman prediction of the motor system.
<img width="882" height="671" alt="image" src="https://github.com/user-attachments/assets/fde473ce-5b72-4be0-ae6b-4cae9ea9c0c6" />


# Referrence
The algorithm and some of the codes comes from the article: 

Korda M, Mezić I. Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control[J]. Automatica, 2018, 93: 149-160.

This code fully leverages Python's matrix parallel computing capabilities to further boost computational speed.
