from buffer_create import *
import torch
from env import *
from tqdm import trange
from scipy.stats import truncnorm



if __name__ == '__main__':
#we present data of three representative patients.

    data_dic ={'P0005': {'tumor': [1134.0, 750.0, 750.0],
  'cd8': [0.2, 0.4679, 0.6111],
  'cd8_raw': 0.0,
  'mu': 0.35,
  'a': 0.0041,
  'sup': 0,
  'anti': 0},
 'P0007': {'tumor': [1505.0, 1056.0, 462.0],
  'cd8': [0.317, 0.4287, 0.4343],
  'cd8_raw': 1.0,
  'mu': 0.9,
  'a': 0.0014,
  'sup': 0,
  'anti': 0},
 'P0023': {'tumor': [1080.0, 896.0, 896.0],
  'cd8': [0.4, 0.8181, 0.4845],
  'cd8_raw': 2.0,
  'mu': 0.0,
  'a': 0.0023,
  'sup': 1,
  'anti': 0}}
    T = np.arange(0, 21*1, 1)
    T = torch.tensor(T)
    env_=env()
    system, params = env_.make_system()
    optim = torch.optim.Adam([params["j1"], params["ml"], params["at"]], lr=0.0002)

    data_dic_conv={}

    for id_ in data_dic.keys():
        state_l = []
        state_next_l = []
        data_dic_conv[id_]={}

        for i in range(len(data_dic[id_]['tumor'])-1):
            tumor=data_dic[id_]['tumor'][i]
            cd8=data_dic[id_]['cd8'][i]
            tumor_next = data_dic[id_]['tumor'][i+1]
            cd8_next= data_dic[id_]['cd8'][i+1]
            state_l.append(torch.tensor([tumor,cd8,1]))

            state_next_l.append(torch.tensor([tumor_next,cd8_next,0]))
        data_dic_conv[id_]['state']=state_l
        data_dic_conv[id_]['state_n']=state_next_l
    key_l=list(data_dic.keys())

    # key_l=['P0005', 'P0007']
    epoch_=1
    for k in trange(200):
        optim.zero_grad()
        loss_l= torch.zeros([len(data_dic.keys())*epoch_])
        for epoch in range(epoch_):
            for i in range(len(key_l)):
                id_= key_l[i]

                x0=data_dic_conv[id_]['state'][0]
                a = data_dic[id_]['a']
                mu = data_dic[id_]['mu']
                anti = data_dic[id_]['anti']
                sup=data_dic[id_]['sup']

                env_.patient_biomarker(a, mu, anti,sup)
                v_init = x0
                v_adjust = torch.tensor([1, 0.2, 0])
                for j in range(len(data_dic[id_]['tumor']) - 1):

                    out = forward_pass(x0, T, system)
                    x0 = out[-1]
                    x0[2] = 1
                    out = forward_pass(x0, T, system)

                    xpred = out[-1]
                    xture=data_dic_conv[id_]['state_n'][j]
                    print(id_,xpred , xture)
                    x0 = out[-1]
                    x0[2]=1

                    if j==0:
                        error = (xpred-xture).unsqueeze(dim=0)
                        error_=((xpred - xture) * v_adjust / v_init).unsqueeze(dim=0)
                    else:
                        error=torch.cat((error,(xpred-xture).unsqueeze(dim=0)),dim=0)
                        error_ = torch.cat((error_,((xpred - xture) * v_adjust / v_init).unsqueeze(dim=0)),dim=0)

                if epoch==epoch_-1:
                    if i ==0:
                        error_all =error
                    else:
                        error_all=torch.cat((error_all,error),dim=0)
                print(error_)
                loss_l[len(data_dic.keys())*epoch+i]=torch.mean(torch.abs(error_))

        # A = np.diag(torch.var(error_all, dim=0).detach().numpy())

        print(torch.var(error_all, dim=0))
        print(torch.mean(loss_l))
        torch.mean(loss_l).backward()

        print(loss_l)
        optim.step()
        clip_weight(params)
        print(k,params)