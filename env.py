from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from functools import partial
import torch.nn as nn

def _rk4_step(fun, yk, tk, h):
    k1 = fun(yk, tk)
    k2 = fun(yk + h / 2 * k1, tk + h / 2)
    k3 = fun(yk + h / 2 * k2, tk + h / 2)
    k4 = fun(yk + h * k3, tk + h)

    yk_next = yk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return yk_next


def rk4(fun, y0, t, retain_grad=False):
    y = []

    h = t[1] - t[0]
    yk = y0
    y.append(yk)

    for i in range(1, len(t)):
        yknext = _rk4_step(fun, yk, t[i - 1], h)
        yk = yknext

        if retain_grad:
            yk.retain_grad()

        y.append(yk)

    return y

def forward_pass(x0, T, system):
    out = rk4(system, x0, T, retain_grad=True)

    return out


def backward_pass(xpred, xdata):
    loss = torch.mean(torch.square(xpred - xdata))

    loss.backward()

    return loss

def clip_weight(params):
    for key in params.keys():
        if params[key]<0:
            print(key)
            params[key].data.fill_(0)


class env():
    def __init__(self):
        self.e = math.e
        self.L_max=1
        self.T_max=1.2e4
        self.k=0.15
        self.relu=nn.ReLU()
        self.a=0.0032
        self.b=2e-4
        self.mu = 0.02
        self.h=0.25


    def patient_biomarker(self,a,mu,anti,sup):
        self.a=a
        self.mu=mu
        self.anti=anti
        if sup==1:
            self.b=2e-5
        else:
            self.b=2e-4

    def make_system(self):
        def system_linear(x,t, params):
            j1,ml,at= params["j1"],params['ml'],params['at']

            mu = self.mu * (1 - x[2])
            # Dl=(b*T)/(1+b*h*T)*Lr

            Lr=x[1]*(1-mu)*self.k

            DlT = self.b * (x[0] * Lr / (1 + self.b * self.h * x[0]))* x[0]

            dx1dt = self.relu((self.a-self.anti*at) *(1 -x[0] / self.T_max)* x[0]) - DlT

            dx2dt =self.relu((DlT/(10+DlT))*j1*(1-x[1]/self.L_max))-ml*x[1]

            dx3dt = -0.035*x[2]
            return torch.cat([dx1dt, dx2dt, dx3dt.unsqueeze(dim=0)] ,dim=0)

        j1 = torch.nn.Parameter(torch.tensor([0.02]))
        ml = torch.nn.Parameter(torch.tensor([0.003]))
        at = torch.nn.Parameter(torch.tensor([0.002]))

        params = {"j1": j1, "ml": ml,'at':at}

        system = partial(system_linear, params=params)

        return system, params
    def make_system_s(self,params):
        def system_linear(x,t, params):
            j1,ml,at=  params["j1"],params['ml'],params['at']
            mu = self.mu * (1 - x[2])
            # Dl=(b*T)/(1+b*h*T)*Lr
            Lr = x[1] * (1 - mu) * self.k
            DlT = self.b * x[0] * Lr / (1 + self.b * self.h * x[0]) * x[0]

            dx1dt = self.relu((self.a - self.anti * at) * (1 - x[0] / self.T_max) * x[0]) - DlT

            dx2dt = self.relu((DlT / (10 + DlT)) * j1 * (1 - x[1] / self.L_max)) - ml * x[1]

            dx3dt = -0.035 * x[2]



            return torch.cat([dx1dt, dx2dt, dx3dt.unsqueeze(dim=0)] ,dim=0)



        system = partial(system_linear, params=params)

        return system