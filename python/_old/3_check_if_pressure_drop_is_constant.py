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

def orifice_discharge(P1, P2, d_orifice, Cd, rho, pipe_pressure_loss):
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
    print('P1, P2, dP,',P1, P2,dP)
    if abs(dP)-pipe_pressure_loss < 0:
        if abs(abs(dP)-pipe_pressure_loss) < 1e-3:
            # オリフィス前後の圧力差と配管の圧力損失が等しい場合、流体の移動は停止
            return 0
        elif abs(dP) - pipe_pressure_loss < 0:
            # 配管での圧力損失が流体の逆流を誘起する場合
            print(colored('Reverse flow induced by pipe pressure loss','red'))
            # TODO: 配管が逆流するのでK factorが異なることになる。計算をやり直す必要がある
            dP = -(abs(dP) - pipe_pressure_loss)
            Q = - Cd * A_orifice * np.sqrt(2 * dP / rho)
            return Q
        else:
            dP = dP - pipe_pressure_loss
            # 通常の計算
            Q = Cd * A_orifice * np.sqrt(2 * dP / rho)  # m^3/s
            return Q
    if dP < 0:
        dP = -dP 
        dP = dP - pipe_pressure_loss
        Q = - Cd * A_orifice * np.sqrt(2 * dP / rho)  # m^3/s
    else:
        dP = dP - pipe_pressure_loss
        Q = Cd * A_orifice * np.sqrt(2 * dP / rho)  # m^3/s
    return Q

def pressure_loss(L, D, rho, v, mu, P1, P2, epsilon=0):
    """
    ダルシー・ワイスバッハの式を用いて配管の圧力損失を計算
    
    Parameters:
        L (float): 配管の長さ (m)
        D (float): 配管の内径 (m)
        rho (float): 流体の密度 (kg/m³)
        v (float): 流速 (m/s)
        mu (float): 流体の動粘度 (Pa·s)
        epsilon (float): 配管の粗さ (m)（デフォルト 0：スムーズな管）
    
    Returns:
        float: 圧力損失 ΔP (Pa)
    """
    # 動粘度 ν = μ / ρ
    nu = mu / rho
    # レイノルズ数 Re = ρ v D / μ
    Re = rho * v * D / mu
    
    # 摩擦係数 f の計算
    print("Re:",Re)
    if Re < 2000:
        # 層流領域（Poiseuille Flow）
        print('層流 Poiseuille Flow Re < 2000')
        f = 64 / Re
    else:
        # 乱流領域（Colebrook-White equation を数値解）
        print('乱流')
        def colebrook(f):
            return 1 / np.sqrt(f) + 2.0 * np.log10(epsilon / (3.7 * D) + 2.51 / (Re * np.sqrt(f)))

        f_initial = 0.02  # 初期推定値
        f = opt.fsolve(colebrook, f_initial)[0]
    print('Friction factor f:',f)

    # 圧力損失 ΔP = f * (L/D) * (ρ v² / 2)
    K_forward = 0.5
    K_reverse = 1.0
    if (P1-P2)>0: 
        K = K_forward
    else:
        K = K_reverse
    delta_P = (f*L/D + K) * (rho * v**2 / 2)
    
    return delta_P

def solve_nonlinear_equation(L, D, rho, P1, P2, Cd, mu, x_old):
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
        K_reverse = 1.0
        K_forward = 0.5
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
        x_left = solve_nonlinear_equation(pipe_length, pipe_diameter_1, rho, P1, P2, Cd, mu, x_left)
        x_right = solve_nonlinear_equation(pipe_length, pipe_diameter_2, rho, P2, P3, Cd, mu, x_right)
        Q_left_new = x_left[4]
        Q_right_new = x_right[4]
        dV_left = Q_left_new * dt  # Volume change (m^3)
        dV_right = Q_right_new * dt  # Volume change (m^3)
        dP1 = - (P1 / V1) * dV_left  # Pressure change in tank 1
        dP2 = (P2 / V2) * dV_left - (P2 / V2) * dV_right # Pressure change in tank 2
        dP3 =  (P3 / V3) * dV_right # Pressure change in tank 3
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

plt.xlabel('Time (s)')
plt.ylabel('Pressure drop in orifice (bar)')
plt.title('Pressure drop at orifice')
plt.grid()
N = 40
t =  downsample_to_N(t_values, N)
dP1 = downsample_to_N(P1_values, N)/100e3 - downsample_to_N(P2_values, N)/100e3
dP2 = downsample_to_N(P3_values, N)/100e3 - downsample_to_N(P2_values, N)/100e3 
plt.plot(t, dP1, 'b-')
plt.plot(t, dP2, 'r-')
plt.legend(['ΔP1','ΔP2'])
plt.show()