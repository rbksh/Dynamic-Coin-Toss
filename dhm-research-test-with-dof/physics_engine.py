import numpy as np
from scipy.integrate import odeint

def generate_5dof_kinematics():
    mass, radius, thickness = 0.005, 0.012, 0.002
    I1 = I2 = 0.25 * mass * radius**2 + (1/12) * mass * thickness**2
    I3 = 0.5 * mass * radius**2
    g0, earth_radius, rho0, scale_height = 9.80665, 6371000, 1.225, 8500

    def equations(state, t):
        z, v_z, w1, w2, w3 = state
        g_z = g0 * (earth_radius / (earth_radius + z))**2
        rho_z = rho0 * np.exp(-z / scale_height)
        lin_drag = 0.001 * rho_z
        rot_drag = 0.0005 * rho_z
        
        dz = v_z
        dv_z = -g_z - (lin_drag * v_z**2 * np.sign(v_z) / mass)
        dw1 = ((I2 - I3) * w2 * w3 - rot_drag * w1**2 * np.sign(w1)) / I1
        dw2 = ((I3 - I1) * w3 * w1 - rot_drag * w2**2 * np.sign(w2)) / I2
        dw3 = ((I1 - I2) * w1 * w2 - rot_drag * w3**2 * np.sign(w3)) / I3
        return [dz, dv_z, dw1, dw2, dw3]

    t_flight = np.linspace(0, 0.6, 1000)
    initial_state = [1.5, 2.94, 5.0, 2.0, 38.0]
    solution = odeint(equations, initial_state, t_flight)
    
    noisy_features = solution + np.random.normal(0, 0.02, solution.shape)
    phase = np.cumsum(solution[:, 4]) * (0.6 / 1000)
    precession = np.radians(15)
    ground_truth_z = np.cos(phase) * np.cos(precession) + np.sin(phase) * np.sin(precession) * np.sin(t_flight * 10)
    
    return noisy_features, ground_truth_z
