import numpy as np

from parameters import *
import math

def calculate_cost(time_step,p_fc,pre_p_fc,p_el,pre_p_el,p_bat,e_buy,pre_soc_bat,soc_bat,soc_h2,delta_m,m_fcv):
    carbon_cost,real_carbon_cost = calculate_carbon(e_buy,time_step)
    degrade_cost = calculate_degrade(p_fc,pre_p_fc,p_el,pre_p_el,pre_soc_bat,soc_bat)
    degrade_cost = 0 if degrade_cost < 0 else degrade_cost
    #degrade_cost = 30 if degrade_cost >= 30 else degrade_cost
    degrade_cost = 0.005 * degrade_cost
    carbon_cost = 40 if carbon_cost >= 40 else carbon_cost
    total_cost = -(0.5*carbon_cost + 0.5*degrade_cost)
    e_reward = calculate_e_reward(time_step,p_el,p_bat)
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

def calculate_degrade(power_FC,FC_pre_power,power_EL,EL_pre_power,pre_soc,bat_soc):

    degrade_FC_cost = calculate_FC_degrade(power_FC, FC_pre_power)
    degrade_FC_cost = 0
    degrade_EL_cost = calculate_EL_degrade(power_EL, EL_pre_power)
    degrade_bat_cost = calculate_bat_degrade(pre_soc,bat_soc)
    total_degrade_cost = degrade_FC_cost + degrade_EL_cost + degrade_bat_cost
    print("degrade_EL_cost is:",degrade_EL_cost)
    print("degrade_bat_cost is:",degrade_bat_cost)
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
    C_EL_var = zeta_EL_var*C_EL*abs(power_EL-EL_pre_power)/V_EL_eol
    C_EL_degrade = C_EL_op + C_EL_s+C_EL_var

    return C_EL_degrade

def calculate_bat_degrade(previous_soc,bat_soc):
    # 修改电池的计算结果,c_bat是电池容量，单位是kwh
    F_pre = 0.5*(1/calculate_N_cycles(1)-1/calculate_N_cycles(previous_soc))
    F_now = 0.5*(1/calculate_N_cycles(1)-1/calculate_N_cycles(bat_soc))
    L_loss = abs(F_now-F_pre)*c_bat*C_inv_bat

    return L_loss

def calculate_N_cycles(DOD):
    N_life_DOD = 21870*np.exp(-1.957*DOD)
    return N_life_DOD

def calculate_battery_soc_restriction_penalty(soc_bat):
    bat_soc_reward = 30 if 0.2 < soc_bat < 0.8 else -30
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
def calculate_e_reward(time_step,p_el,p_bat):
    if 0 <= time_step <= 9 or 17 <= time_step <= 23:
        e_reward = p_el/n_EL*20
    else:
        e_reward = 0
    if 0 <= time_step <= 9 or 21 <= time_step <= 23:
        e_reward_bat = - p_bat/battery_charge_power*10
    else:
        e_reward_bat = p_bat/battery_charge_power*15
    return e_reward+e_reward_bat
def calculate_delta_m_reward(delta_m):
    if -4 < delta_m :
        delta_m_reward = 4
    else:
        delta_m_reward = 0
    return delta_m_reward

