import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
import numpy as np

def downsample_to_N(data, N=20):
    n = len(data)
    if n <= N:
        return data  
    indices = np.linspace(0, n-1, N, dtype=int)
    return [data[i] for i in indices]

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
        Q_left = orifice_discharge(P1, P2, d_orifice_left, Cd, rho)
        Q_right = orifice_discharge(P2, P3, d_orifice_right, Cd, rho)
        # Change of mass in tanks
        dV_left = Q_left * dt  # 体積変化量 (m^3)
        dV_right = Q_right * dt  # 体積変化量 (m^3)
        dP1 = - (P1 / V1) * dV_left  # タンク1の圧力変化
        dP2 = (P2 / V2) * dV_left - (P2 / V2) * dV_right # タンク2の圧力変化
        dP3 =  (P3 / V3) * dV_right
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
dt = 0.01  # Time step (s)
t_max = 40 # Total simulation time (s)

# Run simulation 
t_values, P1_values, P2_values, P3_values = simulate_pressure_variation(P1_init, P2_init, V1, V2, V3,
                                                             d_orifice_left, d_orifice_right,
                                                             Cd, rho, dt, t_max)

# 結果のプロット

REALTIME_PLOT = False
plt.legend(['Tank1','Tank2', 'Tank3'])
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Variation between Tanks')
plt.grid()
if REALTIME_PLOT:
    for i in range(len(t_values)):
        t =  downsample_to_N(t_values[:i])
        P1 = downsample_to_N(P1_values[:i])
        P2 = downsample_to_N(P2_values[:i])
        P3 = downsample_to_N(P3_values[:i])
        plt.cla()
        plt.xlim(0, t_max)
        plt.ylim(0, 900e3)
        plt.plot(t, P1, 'r-')
        plt.plot(t, P2, 'b-')
        plt.plot(t, P3, 'k-')
        plt.pause(0.0000001)
    plt.show()
else:
    N = 40
    t =  downsample_to_N(t_values, N)
    P1 = downsample_to_N(P1_values, N)
    P2 = downsample_to_N(P2_values, N)
    P3 = downsample_to_N(P3_values, N)
    plt.xlim(0, t_max)
    plt.ylim(500e3, 800e3)
    plt.plot(t, P1, 'r-')
    plt.plot(t, P2, 'g-')
    plt.plot(t, P3, 'b-')
    plt.show()