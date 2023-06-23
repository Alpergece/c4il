import numpy as np

# Define the parameters of the inverted pendulum system
m = 0.5     # mass of the pendulum
l = 0.3     # length of the pendulum
g = 9.81    # acceleration due to gravity

# Define the equilibrium point (vertical upright position)
x_eq = np.array([0, np.pi, 0, 0])

# Define the state and input matrices
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, (m*g)/l, 0, 0],
              [0, ((m+1)*g)/(l*m), 0, 0]])

B = np.array([[0],
              [0],
              [-1/(m*l)],
              [1/(m*l)]])

# Define the time step and duration of simulation
dt = 0.01
t_final = 10

# Create a time vector for simulation
t = np.arange(0, t_final, dt)

# Define the initial state
x0 = np.array([0, np.pi + 0.1, 0, 0])

# Define a function to compute the state derivative
def dxdt(x, t, u):
    dx = np.dot(A, x - x_eq) + np.dot(B, u)
    return dx

# Linearize the system using Taylor series expansion
def linearize(x_eq, u_eq):
    A_lin = np.zeros((4, 4))
    B_lin = np.zeros((4, 1))

    dxdt_xi = dxdt(x_eq, 0, u_eq)
    for i in range(4):
        dxdt_xi_flat = dxdt_xi.flatten()
        dxdt_xi_flat[i] += 0.01
        A_lin[:, i] = (dxdt(x_eq + np.array([0, 0, 0.01, 0]), 0, u_eq).flatten() - dxdt_xi_flat) / 0.01
    B_lin = dudt(x_eq, 0, u_eq)

    return A_lin, B_lin


# Compute the linearized system matrices
A_lin, B_lin = linearize(x_eq, 0)

# Print the linearized system matrices
print("Linearized system matrices:")
print("A_lin = ")
print(A_lin)
print("B_lin = ")
print(B_lin)