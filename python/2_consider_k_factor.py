import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
import numpy as np
import scipy.optimize as opt
from termcolor import colored

def downsample_to_N(data, N=20):
    n = len(data)
    if n <= N:
        return np.array(data)
    indices = np.linspace(0, n-1, N, dtype=int)
    return np.array([data[i] for i in indices])

def solve_nonlinear_equation(L, D, rho, P1, P2, Cd, mu, x_old, K_forward, K_reverse):
    def f1(v):
        Re = rho * v * D / mu
        if Re < 2000:
            f = 64 / Re
        else:
            def colebrook(f):
                epsilon = 0.0
                return 1 / np.sqrt(f) + 2.0 * np.log10(epsilon / (3.7 * D) + 2.51 / (Re * np.sqrt(f)))
            f_initial = 0.02  # initial estimate 
            f = opt.fsolve(colebrook, f_initial)[0]
        return f
    def f2(dP):
        if dP < 0:
            K = K_reverse
        else:
            K = K_forward
        return K
    def f3(dP):
        A = np.pi * (D / 2)**2
        if dP < 0:
            Q =  - Cd * A * np.sqrt(- 2 * dP / rho)
        else:  
            Q = Cd * A * np.sqrt(2 * dP / rho)
        return  Q
    def func(x):
        return [x[0] - (x[1] * L / D + x[2]) * rho*(x[5]**2)/2,
                x[1] - f1(x[5]),
                x[2] - f2(P1 - P2),
                x[3] - (P1 - P2) - x[0],
                x[4] - f3(P1 - P2),
                x[5] - x[4] / (np.pi * (D / 2)**2)]
    x_new = opt.fsolve(func, x_old) 
    return x_new

def simulate_pressure_variation(P1_init, P2_init, V1, V2, V3, d_orifice_left, d_orifice_right, Cd, rho, dt, t_max):
    """
    Simulation of pressure fluctuations in two tanks connected via an orifice
    
    P1_init: initial pressure tank 1 (Pa)
    P2_init: initial pressure tank 2 (Pa)
    P3_init: initial pressure tank 3 (Pa)
    V1: Volume of tank 1 (m^3)
    V2: Volume of tank 2 (m^3)
    V3: Volume of tank 3 (m^3)
    d_orifice_left: Diameter of orifice 1 (m)
    d_orifice_right: Diameter of orifice 2 (m)
    Cd: Discharge coefficient of orifice
    rho: Density of fluid (kg/m^3)
    dt: Time step (s)
    t_max: Total simulation time (s)
    """
    t_values = np.arange(0, t_max, dt)
    P1_values = [P1_init]
    P2_values = [P2_init]
    P3_values = [P3_init]
    
    P1 = P1_init
    P2 = P2_init
    P3 = P3_init

    x_left = [1,1,1,1,1,1] # solution vector initialization
    x_right = [1,1,1,1,1,1] # solution vector initialization
    
    for _ in t_values[1:]:
        print("##########")
        pipe_length = 1 #[m]
        pipe_diameter_1 = 0.05 #[m]
        pipe_diameter_2 = 0.025 #[m]
        mu = 1.882e-5# viscosity of air [Pa·s]
        x_left = solve_nonlinear_equation(pipe_length, pipe_diameter_1, rho, P1, P2, Cd, mu, x_left, 0.5, 1.0)
        x_right = solve_nonlinear_equation(pipe_length, pipe_diameter_2, rho, P2, P3, Cd, mu, x_right, 0.5, 1.0)
        Q_left_new = x_left[4]
        Q_right_new = x_right[4]
        dV_left = Q_left_new * dt  # Volume change (m^3)
        dV_right = Q_right_new * dt  # Volume change (m^3)
        dP1 = 0.0                  - (P1 / V1) * dV_left  # Pressure change in tank 1
        dP2 = (P2 / V2) * dV_left  - (P2 / V2) * dV_right # Pressure change in tank 2
        dP3 = (P3 / V3) * dV_right - 0.0 # Pressure change in tank 3
        print('Pressure changes in tanks[Pa]:',dP1,dP2,dP3)
        P1 += dP1
        P2 += dP2
        P3 += dP3
        # print(P1,P2,P3)

        P1_values.append(P1)
        P2_values.append(P2)
        P3_values.append(P3)
    
    return t_values, P1_values, P2_values, P3_values

# Initial condition 
P1_init = 675e3 # initial pressure in tank 1(Pa)
P2_init = 550e3 # initial pressure in tank 2(Pa)
P3_init = 750e3 # initial pressure in tank 3(Pa)
V1 = 50.0  # Volume of tank 1 (m^3)
V2 = 40.0  # Volume of tank 2 (m^3)
V3 = 30.0  # Volume of tank 3 (m^3)
d_orifice_left = 0.05  # Orifice diameter (m) 
d_orifice_right = 0.025  # Orifice diameter (m)
Cd = 1.0  # Discharge coefficient of orifice 
rho = 1.293 # Density of fluid (kg/m^3)
dt = 0.1  # Time step (s)
t_max = 125 # Total simulation time (s)

# Run simulation 
t_values, P1_values, P2_values, P3_values = simulate_pressure_variation(P1_init, P2_init, V1, V2, V3,
                                                             d_orifice_left, d_orifice_right,
                                                             Cd, rho, dt, t_max)

# Plot of the result
REALTIME_PLOT = False
# REALTIME_PLOT = True 
plt.legend(['Tank1','Tank2', 'Tank3'])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (kPa)')
plt.title('Pressure Variation between Tanks')
plt.grid()
if REALTIME_PLOT:
    for i in range(len(t_values)):
        t =  downsample_to_N(t_values[:i])
        P1 = downsample_to_N(P1_values[:i])/1e3
        P2 = downsample_to_N(P2_values[:i])/1e3
        P3 = downsample_to_N(P3_values[:i])/1e3
        plt.cla()
        plt.legend(['Tank1','Tank2', 'Tank3'])
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (kPa)')
        plt.title('Pressure Variation between Tanks')
        plt.grid()
        plt.xlim(0, t_max)
        plt.ylim(500, 900)
        plt.plot(t, P1, 'b-')
        plt.plot(t, P2, 'r-')
        plt.plot(t, P3, 'g-')
        plt.pause(0.0000001)
    plt.show()
else:
    N = 40
    t =  downsample_to_N(t_values, N)
    P1 = downsample_to_N(P1_values, N)/1e3
    P2 = downsample_to_N(P2_values, N)/1e3
    P3 = downsample_to_N(P3_values, N)/1e3
    plt.xlim(0, t_max)
    plt.ylim(500, 800)
    plt.plot(t, P1, 'b-')
    plt.plot(t, P2, 'r-')
    plt.plot(t, P3, 'g-')
    plt.legend(['Tank1','Tank2', 'Tank3'])
    plt.show()