import random

import numpy as np

from parameters import *
from reward_function import calculate_cost
import torch

class EnergySystem():
    def __init__(self,initial_battery_soc:float,initial_h2_soc:float,initial_fc_power:float,initial_el_power:float,initial_bat_power:float,flag:str):
        #basic initialize
        self.freq = freq
        self.r = r
        self.life = life
        self.lambda_inv = lambda_inv
        self.P_price = P_price
        self.time_step = time_step
        self.cost_history = []
        self.flag = flag
        self.action_space_low = np.array([-battery_charge_power,0,0])
        self.action_space_high = np.array([battery_charge_power,n_FC,n_EL])

        #battery initialize
        self.c_bat = c_bat
        self.battery_charge_power = battery_charge_power
        self.battery_discharge_power = battery_discharge_power
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.C_inv_bat = C_inv_bat
        self.C_bat = C_bat
        self.bat_soc_history = [initial_battery_soc]
        self.bat_power_history = [initial_bat_power]

        #electricity initialize
        self.c_G_buy = c_G_buy
        self.p_load_prediction_history = p_load_prediction_history
        self.p_load_reality_history = p_load_reality_history
        self.EL_power_history = [initial_el_power]

        #pv initialize
        self.n_pv = n_pv
        self.G_STC = G_STC
        self.co2_pv = co2_pv
        self.C_inv_pv = C_inv_pv
        self.solar_prediction_history = solar_prediction_history
        self.solar_realty_history = solar_realty_history

        #wt initialize
        self.n_wt = n_wt
        self.V_cut_in = V_cut_in
        self.V_r = V_r
        self.V_cut_out = V_cut_out
        self.co2_wt = co2_wt
        self.C_wt = C_wt
        self.windspeed_prediction_history = windspeed_prediction_history
        self.windspeed_reality_history = windspeed_reality_history

        #fuel cell initialize
        self.n_FC = n_FC
        self.C_inv_FC = C_inv_FC
        self.C_FC = C_FC
        self.FC_power_history = [initial_fc_power]
        self.pre_p_FC = initial_fc_power

        #electrolyser initialize
        self.eta_EL = eta_EL
        self.n_FC = n_EL
        self.C_inv_EL = C_inv_EL
        self.C_EL = C_EL
        self.pre_p_EL = initial_el_power

        #h2 storage tank initialize
        self.V_h2 = V_h2
        self.R = R
        self.T = T
        self.b = b
        self.h2_pressure_max = h2_pressure_max
        self.H2_soc_history = [initial_h2_soc]
        self.h2_soc_history = []
        self.fc_power_history = []
        self.el_power_history = []
        """
        Notice: if we need to train our model, our model could not know the realistic conditions but to make 
            judgements accroding to the results of Informer's prediction
        """
        if self.flag == 'train':
            self.pv_power_history = pv_power_prediction_history
            self.p_load_history = p_load_prediction_history
            self.wt_power_history = wt_prediction_power_history
            self.fcv_h2_history = np.array(fcv_h2_prediction_m_history)
            self.ev_charge_history = ev_charge_predicted_history
        elif self.flag == 'test':
            self.pv_power_history = pv_power_reality_history
            self.p_load_history = p_load_reality_history
            self.wt_power_history= wt_reality_power_history
            self.fcv_h2_history = np.array(fcv_h2_prediction_m_history)/2
            self.ev_charge_history = ev_charge_predicted_history
        self.reset()
    def step(self,action):
        soc_bat,soc_h2,p_load, pv_power,wt_power,m_fcvs,pre_fc_power,pre_el_power,time_step = self.state

        #soc_bat = round(soc_bat,4)
        #soc_h2 = round(soc_h2,4)
        #action = action[0]
        p_load = p_load * max(self.p_load_history)
        pv_power = pv_power * max(self.pv_power_history)
        wt_power = wt_power * max(self.wt_power_history)
        m_fcvs  = m_fcvs * max(self.fcv_h2_history)
        pre_fc_power = pre_fc_power*n_FC
        pre_el_power = pre_el_power*n_EL
        p_bat, p_FC, p_EL = action[0],action[1],action[2]
        p_bat = battery_charge_power*p_bat
        #p_FC = n_FC*(p_FC+1)/2
        p_FC = 0
        p_EL = n_EL*(p_EL+1)/2
        p_bat = min(soc_bat*c_bat,p_bat) if p_bat > 0 else p_bat # 考虑电池没有足够的电输出
        p_bat = max(-(1-soc_bat)*c_bat,p_bat) if p_bat<0 else p_bat # 考虑电池充满以后的情况
        p_load = p_load
        print("========================================================")
        print("This is No.{} time step".format(self.time_step))
        print("SOC of battery:{}    SOC of H2:{}".format(soc_bat,soc_h2))
        print("bat_power:{}, fc_power:{}, el_power:{}".format(p_bat,p_FC,p_EL))
        #p_EL = p_EL.clip(0,n_EL)
        #p_FC = p_h2 if p_h2 > 0 else 0
        #p_EL = abs(p_h2) if p_h2 < 0 else 0
        self.el_power_history.append(p_EL)
        self.fc_power_history.append(p_FC)

        e_buy =-(pv_power + wt_power + p_FC + p_bat - p_EL - p_load-self.ev_charge_history[self.time_step])

        print("e_buy:{}".format(e_buy))
        pre_soc_bat = soc_bat
        soc_bat = soc_bat - (eta_charge * p_bat * freq)/self.c_bat
        soc_bat = round(soc_bat,4)
        if soc_bat > 1:
            soc_bat = 1
        elif soc_bat < 0:
            soc_bat = 0
        soc_h2,delta_m= self.calculate_new_h2_soc(p_FC,p_EL,m_fcvs)
        soc_h2 = round(soc_h2,4)
        self.h2_soc_history.append(soc_h2)
        new_p_load = self.p_load_history[self.time_step+1] if self.time_step!= 23 else self.p_load_history[0]
        new_pv_power = self.pv_power_history[self.time_step+1] if self.time_step!= 23 else self.pv_power_history[0]
        new_wt_power = self.wt_power_history[self.time_step+1] if self.time_step!=23 else self.wt_power_history[0]
        new_m_fcvs = self.fcv_h2_history[self.time_step+1] if self.time_step != 23 else self.fcv_h2_history[0]
        costs,real_carbon_cost,degrade_cost,sum_of_cost = calculate_cost(self.time_step,p_FC,pre_fc_power,p_EL,pre_el_power,p_bat,e_buy,pre_soc_bat,soc_bat,soc_h2,delta_m,m_fcvs)
        pre_fc_power = p_FC
        pre_el_power = p_EL
        self.state = [soc_bat,soc_h2,new_p_load/max(self.p_load_history)
            ,new_pv_power/max(self.pv_power_history),new_wt_power/max(self.wt_power_history),
                      new_m_fcvs/max(self.fcv_h2_history),pre_fc_power/n_FC,pre_el_power/n_EL,self.time_step/23]
        print(self.state)
        print(costs)
        self.time_step += 1
        done = 1 if self.time_step == 24 else 0
        self.pre_p_FC = p_FC
        self.pre_p_EL = p_EL
        return self.state,costs,done,real_carbon_cost,degrade_cost,sum_of_cost

    def test_step(self,action):
        soc_bat,soc_h2,p_load, pv_power,wt_power,m_fcvs,pre_fc_power,pre_el_power,time_step = self.state
        #soc_bat = round(soc_bat,4)
        #soc_h2 = round(soc_h2,4)
        #action = action[0]
        p_load = p_load * max(self.p_load_history)
        pv_power = pv_power * max(self.pv_power_history)
        wt_power = wt_power * max(self.wt_power_history)
        m_fcvs  = m_fcvs * max(self.fcv_h2_history)
        pre_fc_power = pre_fc_power*n_FC
        pre_el_power = pre_el_power*n_EL
        p_bat, p_FC, p_EL = action[0],action[1],action[2]
        p_bat = battery_charge_power*p_bat
        p_FC = n_FC*(p_FC+1)/2
        p_FC = 0
        p_EL = n_EL*(p_EL+1)/2
        p_bat = min(soc_bat*c_bat,p_bat) if p_bat > 0 else p_bat # 考虑电池没有足够的电输出
        p_bat = max(-(1-soc_bat)*c_bat,p_bat) if p_bat<0 else p_bat # 考虑电池充满以后的情况
        p_load = p_load
        print("========================================================")
        print("This is No.{} time step".format(self.time_step))
        print("SOC of battery:{}    SOC of H2:{}".format(soc_bat,soc_h2))
        print("bat_power:{}, fc_power:{}, el_power:{}".format(p_bat,p_FC,p_EL))
        #p_EL = p_EL.clip(0,n_EL)
        #p_FC = p_h2 if p_h2 > 0 else 0
        #p_EL = abs(p_h2) if p_h2 < 0 else 0
        self.el_power_history.append(p_EL)
        self.fc_power_history.append(p_FC)

        e_buy =-(pv_power + wt_power + p_FC + p_bat - p_EL - p_load)

        print("e_buy:{}".format(e_buy))
        soc_bat = soc_bat - (eta_charge * p_bat * freq)/self.c_bat
        soc_bat = round(soc_bat,4)
        if soc_bat > 1:
            soc_bat = 1
        elif soc_bat < 0:
            soc_bat = 0
        soc_h2,delta_m= self.calculate_new_h2_soc(p_FC,p_EL,m_fcvs)
        soc_h2 = round(soc_h2,4)
        self.h2_soc_history.append(soc_h2)
        new_p_load = self.p_load_history[self.time_step+1] if self.time_step!= 23 else self.p_load_history[0]
        new_pv_power = self.pv_power_history[self.time_step+1] if self.time_step!= 23 else self.pv_power_history[0]
        new_wt_power = self.wt_power_history[self.time_step+1] if self.time_step!=23 else self.wt_power_history[0]
        new_m_fcvs = self.fcv_h2_history[self.time_step+1] if self.time_step != 23 else self.fcv_h2_history[0]
        costs,real_carbon_cost,degrade_cost,sum_of_cost = calculate_cost(self.time_step,pv_power,wt_power,p_FC,pre_fc_power,p_EL,pre_el_power,p_bat,e_buy,soc_bat,soc_h2,delta_m,m_fcvs)
        pre_fc_power = p_FC
        pre_el_power = p_EL
        self.state = [soc_bat,soc_h2,new_p_load/max(self.p_load_history)
            ,new_pv_power/max(self.pv_power_history),new_wt_power/max(self.wt_power_history),
                      new_m_fcvs/max(self.fcv_h2_history),pre_fc_power/n_FC,pre_el_power/n_EL,self.time_step/23]
        print(self.state)
        print(costs)
        self.time_step += 1
        done = 1 if self.time_step == 24 else 0
        self.pre_p_FC = p_FC
        self.pre_p_EL = p_EL
        return self.state,costs,done,e_buy,real_carbon_cost,degrade_cost,sum_of_cost
    def test_reset(self):
        pass
    def reset(self):
        self.time_step = 0
        initial_bat_soc = round(random.uniform(0.2,0.35),4)
        initial_h2_soc = round(random.uniform(0.45,0.5),4)
        #initial_bat_soc = 0.33
        #initial_h2_soc = 0.34
        self.bat_soc_history = [initial_bat_soc]
        self.H2_soc_history = [initial_h2_soc]
        #self.bat_soc_history = [0.3511]
        #self.H2_soc_history = [0.45]
        initial_bat_power = 10
        initial_fc_power = 10
        initial_el_power = 100
        self.fc_power_history = [initial_fc_power]
        self.el_power_history = [initial_el_power]
        self.bat_power_history = [initial_bat_power]
        self.cost_history = []
        self.h2_soc_history = []
        self.m = self.V_h2 / (self.b + self.R * self.T /( self.h2_pressure_max * (self.H2_soc_history[0])))
        self.state = [self.bat_soc_history[0], self.H2_soc_history[0], self.p_load_history[0]/max(self.p_load_history),
                      self.pv_power_history[0]/max(self.pv_power_history), self.wt_power_history[0]/max(self.wt_power_history),
                      self.fcv_h2_history[0]/max(self.fcv_h2_history),self.fc_power_history[-1]/n_FC,self.el_power_history[-1]/n_EL,self.time_step/23]
        return self.state

    def calculate_new_h2_soc(self,p_fc,p_el,m_fcvs):
        # p_h2 donates the fuel cell power - electrolyser power
        b = self.b
        V = self.V_h2
        eta_EL = self.eta_EL

        R = self.R
        T = self.T
        freq = self.freq
        # 3600 represent 1 Wh = 3600 J
        delta_m = 3600*0.002*(freq * eta_EL * p_el / LHV_h2-freq*p_fc/(eta_FC*LHV_h2)) - m_fcvs
        print("delta_m:{}".format(delta_m))
        self.m += delta_m
        self.soc_h2 = R*T/((V/self.m)-b)/h2_pressure_max
        if self.soc_h2 > 1:
            self.soc_h2 = 1
        elif self.soc_h2 <0:
            self.soc_h2 = 0

        return self.soc_h2, delta_m