import numpy as np
from mushroom_rl.environments import InvertedPendulum
from mushroom_rl.solvers import lqr as LQR_solver
from mushroom_rl.core import Core

import gym
from gym.spaces import Box, Discrete

# Define observation space
obs_low = np.array([-np.inf, -np.inf, -8, -np.inf])
obs_high = np.array([np.inf, np.inf, 8, np.inf])
observation_space = Box(low=-np.inf, high=np.inf, shape=(1,))
action_space = Box(low=-1., high=1., shape=(1,))

# Create an instance of the environment
env = InvertedPendulum()
state = env.reset()

# Define an action to take
action = np.array([0.5])

# Call the step function
next_state, reward, done, info = env.step(action)

# Run the environment for 1000 steps
core = Core(env, None)
# core.evaluate(n_steps=1000)


def compute_linearized_system( x_ref):
    
    """
    Compute the linearized system matrices around a given operating point.

    Args:
        env (gym.Env): environment object.
        x_ref (np.ndarray): state around which to linearize the system.

    Returns:
        A (np.ndarray): state matrix.
        B (np.ndarray): input matrix.
    """

    # Compute the jacobian of the dynamics with respect to the state
    f_x = np.zeros((1, 1))
    for i in range(1):
        delta = np.zeros(1)
        delta[i] = 1e-6
        f_x[:, i] = ((x_ref + delta)[0] - (x_ref - delta)[0]) / (2.0 * 1e-6)

    # Compute the jacobian of the dynamics with respect to the input
    f_u = np.zeros((observation_space.shape[0], action_space.shape[0]))
    for i in range(action_space.shape[0]):
        delta = np.zeros(action_space.shape[0])
        delta[i] = 1e-6
        f_u[:, i] = ((x_ref + delta)[0] - (x_ref - delta)[0]) / (2.0 * 1e-6)

    # Compute the linearized system matrices
    A = f_x
    B = f_u

    return A, B


# Define the cost function parameters
Q = np.diag([1, 1, 0.1, 0.1]) # state penalty matrix
R = np.diag([0.1]) # control penalty matrix
x_goal = np.array([0, 0])
# cost function
'''
where x is the state, u is the action, Q is a diagonal matrix of state weights, R is a diagonal matrix of action weights, and x_goal is the target state. 
Q matrix should be chosen to encourage the pendulum to reach the target state and stabilize at that position. 
The diagonal elements of the R matrix should be chosen to discourage large and rapid movements of the pendulum.
'''
def time_varying_cost(x, u, t, x_goal):
    if t < horizon / 2:
        Q = np.diag([1, 1, 0.1, 0.1])
    else:
        Q = np.diag([1, 1, 1, 1])

    return (x - x_goal).T @ Q @ (x - x_goal) + u.T @ R @ u



# Define the LQR algorithm
def lqr_algorithm(env, horizon, Q, R):
    A, B = compute_linearized_system(np.array([0, np.pi, 0, 0]))
    lqr_solver = LQR_solver(A, B, time_varying_cost, horizon)
    K = lqr_solver.get_gain()
    return K


# Set the environment parameters
'''
the dynamics of the system can be obtained using the inverted_pendulum module of mushroom-rl.
'''

horizon = 50
mdp = InvertedPendulum()


# Solve the problem using the LQR algorithm
K = lqr_algorithm(mdp, horizon, Q, R)

# Print the computed gains
print("Computed gains: ", K)