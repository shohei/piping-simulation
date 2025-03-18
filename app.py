import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
import numpy as np

def downsample_to_N(data, N=20):
    n = len(data)
    if n <= N:
        return np.array(data)
    indices = np.linspace(0, n-1, N, dtype=int)
    return np.array([data[i] for i in indices])

def orifice_discharge(P1, P2, d_orifice, Cd, rho):
    """
    Function to compute flow rate through an orifice
    
    P1: Pressure in tank 1 (Pa)
    P2: Pressure in tank 2 (Pa)
    d_orifice: Diameter of orifice (m)
    Cd: Discharge coefficient of orifice
    rho: Density of fluid (kg/m^3)

    Returned value: flow rate Q(m^3/s)
    """
    A_orifice = np.pi * (d_orifice / 2) ** 2  # Area of orifice (m^2)
    dP = P1 - P2  
    if dP < 0:
        dP = -dP 
        Q = - Cd * A_orifice * np.sqrt(2 * dP / rho) 
    else:
        Q = Cd * A_orifice * np.sqrt(2 * dP / rho) 
    return Q

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
    
    for _ in t_values[1:]:
        # Flow rate through orifice (m^3/s)
        # IMPORTANT!! Order of arguments (i.e., P1, P2) affects the formulation of pressure drop calculation (dP) below
        Q_left = orifice_discharge(P1, P2, d_orifice_left, Cd, rho) 
        Q_right = orifice_discharge(P2, P3, d_orifice_right, Cd, rho)
        # Change of mass in tanks
        dV_left = Q_left * dt  # Volume change (m^3)
        dV_right = Q_right * dt  # Volume change (m^3)
        dP1 = - (P1 / V1) * dV_left  # Pressure change in tank 1
        dP2 = (P2 / V2) * dV_left - (P2 / V2) * dV_right # Pressure change in tank 2
        dP3 =  (P3 / V3) * dV_right # Pressure change in tank 3
        P1 += dP1
        P2 += dP2
        P3 += dP3
        print(P1,P2,P3)

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
t_max = 40 # Total simulation time (s)

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
    plt.show()