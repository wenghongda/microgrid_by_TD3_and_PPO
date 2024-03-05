from energy_system import EnergySystem
import numpy as np
import pandas as pd
from parameters import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = EnergySystem(0.5,0.5,10,100,10,'test')

    state = env.reset()

    h2_soc_history = np.zeros(24)
    bat_soc_history = np.zeros(24)
    action_history = np.zeros((24,3))
    degrade_cost_history = np.zeros(24)
    carbon_cost_history = np.zeros(24)
    sum_of_cost_history = np.zeros(24)
    e_buy_history = np.zeros(24)


    print(p_load_reality_history)
    for i in range(24):
        p_fc = 0

        p_bat = 0
        soc_bat = state[0]
        soc_h2 = state[1]
        if 0 <= i <= 8 or  12 <= i <= 16 or  21 <= i <= 23:
            if p_pv[i] > p_load[i]:
                if h2_soc_min< soc_h2 < h2_soc_max:
                    if soc_bat < bat_soc_max:
                        p_el = p_pv[i]-p_load[i]
                        p_bat = -100
                        p_buy = p_bat
                    else:
                        p_bat = 0
                        p_el = p_pv[i]-p_load[i]
                else:
                    if soc_bat < bat_soc_max:
                        p_bat =-(p_pv(i)-p_load(i))
                        p_el = n_EL
                    else:
                        p_cur = p_pv[i]-p_load[i]
            else:
                if soc_h2 < h2_soc_max:
                    if soc_bat < bat_soc_max:
                        p_bat = -100
                        p_el = n_EL
                        p_buy = p_el + p_bat + p_load[i]-p_pv[i]
                    else:
                        p_el = n_EL
                        p_buy = p_el + p_load[i] - p_pv[i]

                else:
                    if soc_bat < bat_soc_max:
                        p_bat = -100
                        p_buy = p_bat+p_load[i]-p_pv[i]
                        p_el = 0
                    else:
                        p_bat = 0
                        p_buy = p_load[i]-p_pv[i]
                        p_el = 0
        else:
            if p_pv[i] > p_load[i]:
                if h2_soc_min < soc_h2 < h2_soc_max:
                    p_el = p_pv[i]-p_load[i]
                else:
                    if soc_bat < bat_soc_max:
                        p_el = n_EL
                        p_bat = p_pv[i]-p_load[i]
                    else:
                        p_cur = p_pv[i]-p_load[i]
                        p_el = n_EL
                        p_fc = 0
                        p_bat = 0
            else:
                if soc_h2 > h2_soc_max:
                    p_el = 0
                else:
                    p_el = n_EL
                    if soc_bat > bat_soc_min:
                        p_bat = min(battery_charge_power,(soc_bat-bat_soc_min)*c_bat)
                        p_el = n_EL
                        p_buy = p_load[i]-p_pv[i]-p_bat+p_el
                    else:
                        p_buy = p_load[i]-p_pv[i]+p_el
        action = np.array([p_bat,p_fc,p_el],dtype=np.float32)
        action_history[i] = action

        next_state, reward, done,e_buy,real_carbon_cost,degrade_cost,sum_of_cost = env.test_step(action)
        degrade_cost_history[i] = degrade_cost
        carbon_cost_history[i] = real_carbon_cost
        sum_of_cost_history[i] = sum_of_cost
        print("sum of cost this day: {}".format(sum_of_cost))
        e_buy_history[i] = e_buy
        state = next_state
    t = np.linspace(0,23,24)

    bat_power_history = np.vstack(action_history[:,0])
    fc_power_history = np.vstack(action_history[:,1])
    el_power_history = np.vstack(action_history[:,2])

    bat_power_history = battery_charge_power*bat_power_history
    fc_power_history = (fc_power_history+1)/2*n_FC
    el_power_history = (el_power_history+1)/2*n_EL
    df = pd.read_csv(r"D:\whd_disertation\record\rule_based\rule_based.csv")
    df['e_buy'] = e_buy_history
    df['pv_power'] = pv_power_reality_history
    df['fc_power'] = fc_power_history
    df['wt_power'] = wt_reality_power_history
    df['bat_power'] = bat_power_history
    df['el_power'] = el_power_history
    df['load_power'] = p_load_reality_history
    df['h2_soc_history'] = h2_soc_history
    df['bat_soc_history'] = bat_soc_history
    df.to_csv(r"D:\whd_disertation\record\rule_based\rule_based.csv",index=False)
    print(h2_soc_history)
    print(bat_soc_history)
    print(len(h2_soc_history))
    print(len(bat_soc_history))
    plt.plot(t,h2_soc_history)
    plt.plot(t,bat_soc_history)
    plt.show()
    print("the total cost of this day is :{}".format(sum_of_cost_history.sum()))