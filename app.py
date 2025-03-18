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
    オリフィスを介した流量を計算する関数
    
    P1: タンク1の圧力 (Pa)
    P2: タンク2の圧力 (Pa)
    d_orifice: オリフィスの直径 (m)
    Cd: オリフィスの流量係数
    rho: 流体の密度 (kg/m^3)
    
    戻り値: 流量 (m^3/s)
    """
    A_orifice = np.pi * (d_orifice / 2) ** 2  # オリフィスの断面積 (m^2)
    dP = max(P1 - P2, 0)  # 圧力差 (負の流れを防ぐため max を使用)
    Q = Cd * A_orifice * np.sqrt(2 * dP / rho)  # オリフィス流量方程式
    return Q

def simulate_pressure_variation(P1_init, P2_init, V1, V2, d_orifice, Cd, rho, dt, t_max):
    """
    オリフィスを介して接続された2つのタンクの圧力変動をシミュレーション
    
    P1_init: タンク1の初期圧力 (Pa)
    P2_init: タンク2の初期圧力 (Pa)
    V1: タンク1の体積 (m^3)
    V2: タンク2の体積 (m^3)
    d_orifice: オリフィスの直径 (m)
    Cd: オリフィスの流量係数
    rho: 流体の密度 (kg/m^3)
    dt: 時間ステップ (s)
    t_max: シミュレーション時間 (s)
    """
    t_values = np.arange(0, t_max, dt)
    P1_values = [P1_init]
    P2_values = [P2_init]
    
    P1 = P1_init
    P2 = P2_init
    
    for _ in t_values[1:]:
        # オリフィスを介した流量 (m^3/s)
        Q = orifice_discharge(P1, P2, d_orifice, Cd, rho)
        # タンク内の質量変化
        dV = Q * dt  # 体積変化量 (m^3)
        dP1 = - (P1 / V1) * dV  # タンク1の圧力変化
        dP2 = (P2 / V2) * dV  # タンク2の圧力変化
        P1 += dP1
        P2 += dP2
        P1_values.append(P1)
        P2_values.append(P2)
    
    return t_values, P1_values, P2_values

# 初期条件
P1_init = 675e3 # タンク1の初期圧力 (Pa)
P2_init = 550e3 # タンク2の初期圧力 (Pa)
V1 = 50.0  # タンク1の体積 (m^3)
V2 = 40.0  # タンク2の体積 (m^3)
d_orifice = 0.05  # オリフィス直径 (m)
Cd = 1.0  # オリフィスの流量係数
rho = 1.293 # 流体密度 (kg/m^3)
dt = 0.01  # 時間ステップ (s)
t_max = 10  # シミュレーション時間 (s)

# シミュレーション実行
t_values, P1_values, P2_values = simulate_pressure_variation(P1_init, P2_init, V1, V2, d_orifice, Cd, rho, dt, t_max)

# 結果のプロット

plt.legend('Tank1','Tank2')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.title('Pressure Variation between Two Tanks')
plt.grid()
for i in range(len(t_values)):
    t =  downsample_to_N(t_values[:i])
    P1 = downsample_to_N(P1_values[:i])
    P2 = downsample_to_N(P2_values[:i])
    plt.cla()
    plt.xlim(0, t_max)
    plt.ylim(0, 700e3)
    plt.plot(t, P1, 'ro')
    plt.plot(t, P2, 'bo')
    plt.pause(0.0000001)
plt.show()