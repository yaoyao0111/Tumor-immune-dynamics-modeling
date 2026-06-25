import numpy as np
import torch
from m1_env import *
from tqdm import trange
from scipy.stats import truncnorm
from tils import *
import copy

if __name__ == '__main__':
#we present data of three represetative patients in pal-based and pem-based cohorts, respectively.
    chemo='pal'
    if chemo=='pal':
        data_dic = {'P0003': {'tumor': [360.0, 220.0, 220.0],
                              'cd8': [0.2, 0.3658, 0.2749],
                              'cd8_raw': 0.0,
                              'mu': 0.0,
                              'a': 0.0059,
                              'sup': 0,
                              'anti': 0},
                    'P0004': {'tumor': [1760.0, 1330.0, 1258.0],
                              'cd8': [0.3479, 0.281, 0.4259],
                              'cd8_raw': 1.3388112,
                              'mu': 0.0,
                              'a': 0.0051,
                              'sup': 1,
                              'anti': 0},
                    'P0009': {'tumor': [2090.0, 312.0, 186.0],
                              'cd8': [0.6252, 0.4267, 0.3731],
                              'cd8_raw': 6.729132,
                              'mu': 0.85,
                              'a': 0.0051,
                              'sup': 0,
                              'anti': 0},
                    }
        r_dic = {'P0003': 0.29,
             'P0004': 0.46,
             'P0009': 0.0
             }

    else:
        data_dic = {'P0001': {'tumor': [3920.0, 3640.0, 0.0],
      'cd8': [0.317, 0.3034, 0.2],
      'cd8_raw': 1.0,
      'mu': 0.7,
      'a': 0.0078,
      'sup': 1,
      'anti': 0},
     'P0011': {'tumor': [2408.0, 1638.0, 1100.0],
      'cd8': [0.5615, 0.3235, 0.4315],
      'cd8_raw': 5.0,
      'mu': 0.01,
      'a': 0.0023,
      'sup': 1,
      'anti': 0},
     'P0014': {'tumor': [234.0, 234.0, 234.0],
      'cd8': [0.3652, 0.3786, 0.3893],
      'cd8_raw': 1.5449389,
      'mu': 0.0,
      'a': 0.0051,
      'sup': 0,
      'anti': 0}
     }
        r_dic={'P0001': 0.0,
         'P0011': 0.38,
         'P0014': 0.78
         }



    T = np.arange(0, 21*1, 1)
    T = torch.tensor(T)
    env_=env_m1()
    system, params = env_.make_system()
    optim = torch.optim.Adam([params["m1"], params["jm1"],params["m1_"]], lr=0.001)
    data_dic_conv={}


    for id_ in data_dic.keys():
        state_l = []
        state_next_l = []
        data_dic_conv[id_] = {}
        try:
            r=r_dic[id_]+0.001
        except:
            r=0.001 

        for i in range(len(data_dic[id_]['tumor']) - 1):
            tumor = data_dic[id_]['tumor'][i]
            cd8 = data_dic[id_]['cd8'][i]
            tumor_next = data_dic[id_]['tumor'][i + 1]
            cd8_next = data_dic[id_]['cd8'][i + 1]
            state_l.append(torch.tensor([tumor*(1-r),tumor*r, cd8, 1,5]))

            state_next_l.append(torch.tensor([tumor_next,0, cd8_next, 0,0]))
        data_dic_conv[id_]['state'] = state_l
        data_dic_conv[id_]['state_n'] = state_next_l
    key_l = list(data_dic.keys())

    epoch_=1

    for k in trange(100):
        optim.zero_grad()
        loss_l= torch.zeros([len(data_dic.keys())*epoch_])
        for epoch in range(epoch_):
            for i in range(len(key_l)):
                id_= key_l[i]
                x0 = data_dic_conv[id_]['state'][0]
                a = data_dic[id_]['a']
                mu = data_dic[id_]['mu']
                anti = data_dic[id_]['anti']
                sup = data_dic[id_]['sup']
                env_.patient_biomarker(a, mu, anti,sup)
                v_init = copy.deepcopy(x0)
                v_init[0]+=v_init[1]
                print(x0)
                v_adjust=torch.tensor([1,0,0.5,0,0])
                for j in range(len(data_dic[id_]['tumor']) - 1):
                    out = forward_pass(x0, T, system)
                    x0 = out[-1]
                    x0[3] = 1
                    x0[4]= 5
                    out = forward_pass(x0, T, system)
                    xpred = out[-1]
                    xture=copy.deepcopy(data_dic_conv[id_]['state_n'][j])

                    x0 = out[-1]

                    x0[3]=1
                    x0[4]=5
                    xture[0] -= xpred[1].detach()
                    print(id_, xpred, xture)
                    if j==0:
                        error =  (xpred - xture).unsqueeze(dim=0)
                        error_ = ((xpred - xture) * v_adjust / v_init).unsqueeze(dim=0)
                    else:
                        error = torch.cat((error, (xpred - xture).unsqueeze(dim=0)), dim=0)
                        error_ = torch.cat((error_, ((xpred - xture) * v_adjust / v_init).unsqueeze(dim=0)), dim=0)

                if epoch==epoch_-1:
                    if i ==0:
                        error_all =error
                    else:
                        error_all=torch.cat((error_all,error),dim=0)
                print(error_)
                loss_l[len(data_dic.keys())*epoch+i]=torch.mean(torch.abs(error_))


        print(torch.var(error_all, dim=0))
        torch.mean(loss_l).backward()

        print(loss_l)
        optim.step()
        clip_weight(params)
        print(k,params)