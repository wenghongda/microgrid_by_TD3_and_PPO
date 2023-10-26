import numpy as np

from parameters import *
import math

def calculate_cost(time_step,pv_power,wt_power,p_fc,pre_p_fc,p_el,pre_p_el,p_bat,e_buy,soc_bat,soc_h2,delta_m,m_fcv):
    carbon_cost = calculate_carbon(e_buy)

    degrade_cost = calculate_degrade(p_fc,pre_p_fc,p_el,pre_p_el,p_bat)
    degrade_cost = 0 if degrade_cost < 0 else degrade_cost
    degrade_cost = sigmoid(degrade_cost)
    carbon_cost = sigmoid(carbon_cost)
    total_cost = -(0.8*carbon_cost + 0.2*degrade_cost)
    power_reward = calculate_power_reward(p_bat,p_fc,p_el,delta_m)
    battery_soc_restriction_penalty = calculate_battery_soc_restriction_penalty(soc_bat)
    h2_soc_restriction_penalty = calculate_h2_soc_restriction_penalty(soc_h2)
    extra_reward = calculate_extra_reward(soc_bat,soc_h2)
    print("the carbon cost is :{}".format(-carbon_cost))
    print("the degrade cost is :{}".format(-degrade_cost))
    print("the total_cost is :{}".format(total_cost))

    print("battery soc restriction penalty :{}".format(battery_soc_restriction_penalty))
    print("h2 soc restriction penalty :{}".format(h2_soc_restriction_penalty))
    print("power reward is :{}".format(power_reward))
    print("extra reward is :{}".format(extra_reward))

    total_cost = extra_reward+total_cost + h2_soc_restriction_penalty+ battery_soc_restriction_penalty
    print("overall reward function: {}".format(total_cost))
    #total_cost = round(total_cost,4)
    return total_cost
def calculate_extra_reward(soc_bat,soc_h2):
    global extra_reward
    if 0.2 < soc_bat < 0.8 and 0.2 < soc_h2 < 0.8:
        extra_reward = 3
    else:
        extra_reward = 0
    return extra_reward



def calculate_carbon(e_buy):
    if e_buy > 0:
        c_co2 = e_buy # electricity bought from main grid; kwh
        P_CO2 = P_price # RMB/Kg
        carbon_cost = P_CO2 * c_G_buy * c_co2 + 0.56*e_buy
    else:
        carbon_cost = 0*e_buy
    return carbon_cost

def calculate_degrade(power_FC,FC_pre_power,power_EL,EL_pre_power,power_bat):

    degrade_FC_cost = calculate_FC_degrade(power_FC, FC_pre_power)
    degrade_EL_cost = calculate_EL_degrade(power_EL, EL_pre_power)
    degrade_bat_cost = calculate_bat_degrade(power_bat)
    total_degrade_cost = degrade_FC_cost + degrade_EL_cost + degrade_bat_cost
    return total_degrade_cost

def calculate_FC_degrade(power_FC, pre_power):
    pre_delta = 1 if pre_power > 0 else 0
    delta_FC = 1 if power_FC > 0 else 0

    delta_FC_low = 1 if power_FC < 0.2 * n_FC else 0
    delta_FC_high = 1 if power_FC > 0.8 * n_FC else 0
    C_FC_low = delta_FC_low * zeta_FC_low * C_FC / V_FC_eol
    C_FC_high = delta_FC_high * zeta_FC_high * C_FC / V_FC_eol
    C_FC_chg = zeta_FC_chg * abs(power_FC - pre_power) * C_FC / (n_FC * V_FC_eol)
    C_FC_s = zeta_FC_s * abs(delta_FC - pre_delta) * C_FC / V_FC_eol
    C_FC_deg = C_FC_low + C_FC_high + C_FC_chg + C_FC_s
    return C_FC_deg

def calculate_EL_degrade(power_EL,EL_pre_power):
    delta_EL = 1 if power_EL > 0 else 0
    pre_delta_EL = 1 if EL_pre_power > 0 else 0
    C_EL_op = delta_EL * zeta_EL_op * C_EL / V_EL_eol
    C_EL_s = zeta_EL_s * C_EL * abs(delta_EL - pre_delta_EL) / V_EL_eol
    C_EL_degrade = C_EL_op + C_EL_s

    return C_EL_degrade

def calculate_bat_degrade(power_bat):
    Q_0 = C_bat * 1000
    N_cycles = calculate_N_cycles(power_bat)
    degrade_bat_cost = C_bat*freq*power_bat/(2*N_cycles*Q_0)
    return degrade_bat_cost

def calculate_N_cycles(power):
    N_cycles = p1*power**6 + p2*power**5+p3*power**4+p4*power**3 + \
                p5 * power**2+p6*power+p7

    return N_cycles
def calculate_delta_m_reward(delta_m):
    if delta_m > -0.5:
        delta_m_reward = 10
    else:
        delta_m_reward = delta_m*0.3
    return delta_m_reward

def calculate_battery_soc_restriction_penalty(soc_bat):
    bat_soc_reward = np.log((np.abs(soc_bat-0.3)+np.abs(soc_bat-0.7))/(0.7-0.3))
    return -bat_soc_reward/2
def calculate_h2_soc_restriction_penalty(soc_h2):
    h2_soc_reward = np.log((np.abs(soc_h2 - 0.3) + np.abs(soc_h2 - 0.7)) /(0.7 - 0.3))
    return -h2_soc_reward/2
def calculate_power_reward(p_bat,p_fc,p_el,delta_m):
    reward = 100 if -25 < p_bat < 25 and 0 < p_fc < n_FC and 0 < p_el < n_EL else 0
    #reward = reward+10 if delta_m > -2 else reward
    return reward

def sigmoid(x):
    return 1/(1+math.exp(-x))