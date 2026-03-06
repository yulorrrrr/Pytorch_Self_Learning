'''
Demontrate weather distribution in 30 days

For β (the weight adjustment coefficient), the larger its value:
    the more the algorithm relies on the exponential moving average,
    and the less it depends on the current local gradient value,
    resulting in a smoother numerical change.
'''

import torch
import matplotlib.pyplot as plt
 
ELEMENT_NUMBER = 30

#Actual avg temperature
def dm01():
    #set a random seed
    torch.manual_seed(0)
    #generate random temperature for 30 days
    temperature = torch.empty(ELEMENT_NUMBER).uniform_(0, 40)
    #print(temperature)

    #plot average temperature
    days = torch.arange(1, ELEMENT_NUMBER+1, 1)
    plt.plot(days, temperature, color='r')
    plt.scatter(days, temperature)
    plt.show()

#Exponentially Weighted Average Temperature
def dm02(beta):
    #set a random seed
    torch.manual_seed(0)
    #generate random temperature for 30 days
    temperature = torch.empty(ELEMENT_NUMBER).uniform_(0, 40)
    #print(temperature)

    exp_weight_avg = []
    #index start from 1
    for idx, temp in enumerate(temperature,1):
        # for the first element, its EWA value equal to itself
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        # for the second element, its EWA value equal to the previous EWA value times 1-beta + current temp 
        #index-2: 2-2 = 0
        new_temp = exp_weight_avg[idx-2] * beta + temp * (1-beta)
        exp_weight_avg.append(new_temp)
    days = torch.arange(1, ELEMENT_NUMBER+1, 1)
    plt.plot(days, exp_weight_avg, color = 'r')
    plt.scatter(days, temperature)
    plt.show()
#test
if __name__ =='__main__':
    dm01()     #don't consider weight
    dm02(0.5)  #consider weight, the smaller the weight is, the steeper the curve is
    dm02(0.9)  #consider weight, the larger the weight is, the smoother the curve is