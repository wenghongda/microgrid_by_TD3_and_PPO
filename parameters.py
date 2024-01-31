#basic parameters
import numpy as np
freq = 1 # the time interval is one hour
r = 0.067
life = 20
lambda_inv = r * (1 + r) ** life / ((1 + r) ** life - 1)
P_price = 0.3 # the price of CO2; RMB/kg
time_step = 0

#battery parameters
bat_soc_min = 0.2
bat_soc_max = 0.7
c_bat = 546 # the capacity of battery is 546; KWH
battery_charge_power = 25 # maximum charge power; KW
battery_discharge_power = 25 # maximum discharge power; KW
eta_charge = 0.8
eta_discharge = 0.8
C_inv_bat = 1000 # investment cost of battery; RMB/KWh
C_bat = lambda_inv * c_bat * C_inv_bat
'''polynomial parameters used to fit the cycle number of battery charges'''
p1 = 0.0005009
p2 = -0.04606
p3 = 1.567
p4 = -22.85
p5 = 107.4
p6 = 119.5
p7 = 2269

#electricity
c_G_buy = 0.928 # the CO2 coefficient for puchasing electricity from the main grid; kg/kWh
p_load_prediction_history = [12.726239, 12.780168, 12.5274, 12.329167, 12.146647, 12.011887, 12.232092,
                                          12.696661, 16.572552, 33.250713, 46.126137, 47.353844, 46.223297, 47.711754,
                                          46.769104, 45.27104, 41.838642, 35.419, 20.959993, 16.705698, 14.00523,
                                          13.36671, 13.150275, 12.992007]
p_load_reality_history = [14.633, 15.497, 14.195, 13.413, 13.69, 12.643, 12.744, 14.669, 21.077, 36.631001,
                                       46.771, 45.914001, 46.325001, 44.990002, 47.618, 50.102001, 44.181, 38.25,
                                       41.775002, 15.261, 10.739, 11.386, 10.645, 11.256]

#photovoltaic panel parameters
n_pv = 350 # the maximum generation power of pv panels; KW
G_STC = 1000 # irradiance at standard test condition (STC); W/m^s
co2_pv = 0.02 # photovolataic CO2 emission coefficient; kg/kWH
C_inv_pv = 7000 # investment cost of pv; RMB/KW
C_pv = lambda_inv * n_pv * C_inv_pv
solar_prediction_history = np.array([0, 0, 0, 3.762482, 16.612305, 47.072647, 118.503197, 205.827347, 449.708496,
                                         706.843201, 812.868591, 928.403625, 934.278748, 874.844299, 762.966187,
                                         661.925293, 483.080627, 339.571838, 134.070404, 52.82074, 27.367065, 15.073425,
                                         0, 0])
solar_realty_history = np.array([0, 0, 0, 0, 0, 1.985, 27.0646, 246.9319, 463.760406, 652.281189, 816.656921,
                                     939.164917, 766.887817, 653.48999, 580.374207, 367.833008, 87.3349, 30.632601,
                                     9.566, 3.8179, 0.9149, 0, 0, 0])
pv_power_prediction_history = solar_prediction_history*n_pv/G_STC
pv_power_reality_history = solar_realty_history*n_pv/G_STC

#wind turbine parameters
n_wt = 329.7 # the maximum generation power of wt; KW
n = 3 # the rate of characteristic curve between v_cut_in and v_r
V_cut_in = 2 # the cut in wind speed of wind turbine; m/s
V_r = 14
V_cut_out = 25 # the cut out wind speed of wind turbine; m/s
co2_wt = 0.01 # wind power CO2 emission coefficient; kg/kWH
C_inv_wt = 15000 # investment cost of wt; RMB/KW
C_wt = lambda_inv * n_wt * C_inv_wt
windspeed_prediction_history = np.array([1.596825, 2.229355, 2.174814, 2.166379, 2.190578, 2.173292, 2.051161,
                                             2.061772, 2.279166, 2.132345, 2.017253, 2.029668, 1.739319, 1.883462,
                                             1.732812, 1.791204, 1.916874, 1.781735, 1.837495, 1.980819, 2.012278,
                                             2.040139, 2.031322, 1.968592])
windspeed_reality_history = np.array([1.3838, 2.3387, 1.8392, 1.3483, 3.4548, 4.458, 1.3544, 1.1686, 1.2487, 1.2541,
                                          1.9477, 1.9603, 2.1216, 1.805, 1.8535, 2.1124, 1.9582, 3.083, 3.4068, 2.661,
                                          2.2088, 1.8512, 1.5547, 0.9575])
wt_prediction_power_history = n_wt*np.array([0, 0.0011257117037422931, 0.000835698067245053,0.0007921219037301383,
                       0.0009180473803088372,0.0008278101910324978,0.00023017931241925643, 0.0002793839138396091,
                       0.0014032713304908428,0.0006197182906304691, 7.632570602545892e-05, 0.00013206259313873017,
                       0, 0, 0, 0, 0, 0, 0, 0, 5.4182144226501195e-05, 0.00017960508240029435, 0.000139539888562133, 0])
wt_reality_power_history = n_wt*np.array([0, 0.0017513015429835514, 0, 0, 0.012147390195391813, 0.029458055523391816, 0,0,
                                  0, 0, 0, 0, 0.000566417180444444,0, 0,
                                  0.000521207088678363,0, 0.007786393562500001, 0.011527885951181288,
                                  0.003962840197733918,0.0010147250677894745, 0, 0, 0])

#fuel cell parameters
zeta_FC_low = 8.622 * 10 ** (-6)  # uV->V
zeta_FC_high = 10 * 10 ** (-6)  # uV->V
zeta_FC_s = 13.79 * 10 ** (-6)  # uV->V
zeta_FC_chg = 0.04185 * 10 ** (-6)  # (uV/kW)->(V/kW)
eta_FC = 0.8 # the efficiency of Fuel cell
n_FC = 65.7 # the maximum generation power of fuel cell; KW
C_inv_FC = 10000 # Fuel cell investment cost; RMB/KW
C_FC = lambda_inv * n_FC * C_inv_FC
V_FC_eol = 1 # 10% * 657 = 65.7 V


#electrolyser parameters
eta_EL = 0.8 # the efficiency of electrolyser
n_EL = 350 # the output power of electrolyser
C_inv_EL = 9000 # the investment cost of electrolyser
C_EL = lambda_inv * n_EL * C_inv_EL
zeta_EL_op = 32 * 10 ** -6  # uV -> V
zeta_EL_s = 30 * 10 ** -6  # uV -> V
V_EL = 1.8  # the standard output voltage of EL is 1.8V
V_EL_eol = 0.2 * V_EL

# EV charge
ev_charge_predicted_history = np.array([47.71, 37.986, 28.91, 21.83,16.8238974720000,13.3781250000000,
                               10.8405210880000,8.72563214400000,6.86628441600002,5.44793615200003,
                               4.94999999999984, 6.01987996800009, 9.30546726400042, 15.2718393360001,
                               24.0279064320000,35.1887499999995, 47.7993972480003, 60.3457761840007,
                               70.8785954560009, 77.2758933120011, 77.6699999999974,71.0646579280027,
                               58.1680439040001,42.4674377760090])
#h2 storage tank parameters
h2_soc_min = 0.2
h2_soc_max = 0.8
LHV_h2 = 240 # the lower heating value of hydrogen (MJ/kmol)
V_h2 = 6 # the volume of hydrogen storage tank; (m^3)
R = 4126.83 # the gas constant of hydrogen; (J/(kg*K))
T = 300 # the temperature of hydrogen tank and it is set to 300 to simplify the calculation; K
b = 7.691*10**(-3) # the correction constant in Abel-Nobel gas equation; 1
h2_pressure_max = 7*10**7 # the maximum pressure that hydrogen tank could contain; Pa
fcv_h2_prediction_m_history = [5.1969, 3.9690687468722854, 3.0766044951753653, 2.3617463825522274,
                               1.854196511300999, 1.7812633167626286, 2.4242970824998995, 3.91359056844557,
                               6.095557679830765, 8.562575437047926, 10.834363934682523, 12.583900449881234,
                               13.771129210055427, 14.603429059112027, 15.348733350835078, 16.11886485961971,
                               16.759744708828233, 16.919378207322392, 16.254451724064953, 14.65018751817482,
                               12.319207197262063, 9.718939226337277, 7.337646589816499, 5.477888029183716]
fcv_h2_reality_m_history = [1.2879260000000001, 1.753453, 3.3051579999999996,5.451722,
                            8.503455,12.020684000000001,15.201718000000001,17.555176,
                            18.667239000000002,19.494815, 20.400015000000003, 21.486193,
                            22.598287,23.270677,22.106906000000002,19.882749,
                            16.624153, 12.718959000000002, 9.408624, 6.951719000000001,
                            4.675854,3.175857, 2.2706880000000003, 2.193095]

#network parameters
log_freq = 5
print_freq = 5
save_model_freq = 10
debug_model_freq = 100
action_min_limitation = np.array([0,0,-battery_charge_power,0])
action_max_limitation = np.array([n_FC,n_EL,battery_charge_power,10000])
N_S = 9
N_A = 3  # Here we think the two dimensions control the soc of battery and hydrogen tank
max_training_iteration = 1000
debug_interval = 500
max_ep_len = 24
lr_actor = 0.0001
lr_critic = 0.0001
gamma = 0.98
lambd = 1
batch_size = 24
epsilon = 0.2
lr_rate = 0.001