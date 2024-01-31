import numpy as np

from parameters import *
import math

def calculate_cost(time_step,pv_power,wt_power,p_fc,pre_p_fc,p_el,pre_p_el,p_bat,e_buy,soc_bat,soc_h2,delta_m,m_fcv):
    carbon_cost,real_carbon_cost = calculate_carbon(e_buy,time_step)
    degrade_cost = calculate_degrade(p_fc,pre_p_fc,p_el,pre_p_el,p_bat)
    degrade_cost = 0 if degrade_cost < 0 else degrade_cost
    carbon_cost = 40 if carbon_cost >= 40 else carbon_cost
    total_cost = -(0.3*carbon_cost + 0.7*degrade_cost)
    e_reward = calculate_e_reward(time_step,p_el)
    final_reward = calculate_final_reward(soc_h2,time_step)
    battery_soc_restriction_penalty = calculate_battery_soc_restriction_penalty(soc_bat)
    h2_soc_restriction_penalty = calculate_h2_soc_restriction_penalty(soc_h2,time_step)
    extra_reward = calculate_delta_m_reward(delta_m)
    print("the carbon cost is :{}".format(-carbon_cost))
    print("the degrade cost is :{}".format(-degrade_cost))
    print("the total_cost is :{}".format(total_cost))

    print("battery soc restriction penalty :{}".format(battery_soc_restriction_penalty))
    print("h2 soc restriction penalty :{}".format(h2_soc_restriction_penalty))
    print("extra reward is :{}".format(extra_reward))

    total_cost = e_reward+ total_cost + h2_soc_restriction_penalty+ battery_soc_restriction_penalty + extra_reward +final_reward
    print("overall reward function: {}".format(total_cost))
    return total_cost,real_carbon_cost,degrade_cost,real_carbon_cost+degrade_cost


def calculate_carbon(e_buy,time_step):
    global real_carbon_cost
    if 12 > time_step > 9 or 16 < time_step < 21:
        price = 0.15
        #price = 0.56*1.5
        real_price=0.56*1.5
    else:
        price = 0.1
        real_price=0.56

    if e_buy > 0:
        c_co2 = e_buy # electricity bought from main grid; kwh
        P_CO2 = P_price # RMB/Kg
        real_carbon_cost = P_CO2 * c_G_buy * e_buy + real_price*e_buy
        carbon_cost = price*e_buy
    else:
        carbon_cost = -10*price*e_buy
        real_carbon_cost = 0
    return carbon_cost,real_carbon_cost

def calculate_degrade(power_FC,FC_pre_power,power_EL,EL_pre_power,power_bat):

    degrade_FC_cost = calculate_FC_degrade(power_FC, FC_pre_power)
    degrade_FC_cost = 0
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
    Q_0 = c_bat
    N_cycles = calculate_N_cycles(power_bat)
    degrade_bat_cost = C_bat*freq*power_bat/(2*N_cycles*Q_0)
    return degrade_bat_cost

def calculate_N_cycles(power):
    N_cycles = p1*power**6 + p2*power**5+p3*power**4+p4*power**3 + \
                p5 * power**2+p6*power+p7

    return N_cycles

def calculate_battery_soc_restriction_penalty(soc_bat):
    bat_soc_reward = 8 if 0.2 < soc_bat < 0.4 else 0
    return bat_soc_reward
def calculate_h2_soc_restriction_penalty(soc_h2,time_step):
    if 0 <= time_step <= 9:
        h2_soc_reward = 8 if 0.45 < soc_h2 < 0.8 else 20*np.log(abs(0.45-soc_h2)+abs(0.8-soc_h2))
    else:
        h2_soc_reward = 8 if 0.45 < soc_h2 < 0.8 else 20*np.log(abs(0.45-soc_h2)+abs(0.8-soc_h2))
    return h2_soc_reward
def calculate_final_reward(soc_h2,time_step):
    if time_step == 23 and 0.45 < soc_h2:
        final_reward = 50
    else:
        final_reward = 0
    return final_reward
def calculate_e_reward(time_step,p_el):
    if 0 <= time_step <= 9 or 21 <= time_step <= 23:
        e_reward = p_el/n_EL*20
    else:
        e_reward = 0
    return e_reward
def calculate_delta_m_reward(delta_m):
    if -4 < delta_m :
        delta_m_reward = 4
    else:
        delta_m_reward = 0
    return delta_m_reward

