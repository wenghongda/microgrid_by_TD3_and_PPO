# microgrid_by_TD3_and_PPO
## The microgrid includes a fuel cell, a electrolyser, a hydrogen storage tank, as well as a battery.
In order to deal with uncertainties such as solar, wind speed, electricity load and hydrogen demand of FCVs, Informer is used to 
achieve a long-term prediction, then PPO and TD3 are used and compared to calculate the optimal energy management strategy of 
this microgrid. 
The objective function of this strategy is to minimize the cost of carbon emission and the equipment degradation including fuel cell,
electrolyser, and the battery.

The training effect of TD3 algorithm is as following:
![image](https://github.com/wenghongda/microgrid_by_TD3_and_PPO/assets/130994851/1f1cd5f7-4716-43c2-8d31-045fb52fbcfd)
The training effect of PPO algorithim is as following:

The effect of (non linear programming) will lately be compared with training effects of PPO and TD3
