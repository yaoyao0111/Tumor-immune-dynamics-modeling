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

class env_m1():
    def __init__(self):
        self.e = math.e

        self.L_max=1
        self.T_max=1.5e4
        self.k=0.15
        self.relu=nn.ReLU()
        self.a=0.0032
        self.b=2e-4
        self.mu = 0.02
        self.h=0.25


        # CD8+T killing effect

        self.l = 1

        self.s = 8.39e-2

        self.gamma_i = 9 * 10 ** (-1)
        self.e = math.e
        self.at=0.0

        self.T_max=1.5e4

        self.relu=nn.ReLU()
        self.j=0.0068
        self.ml=0.0

    def patient_biomarker(self, a, mu, anti,sup):
        self.a = a
        self.mu = mu
        self.anti = anti
        if sup == 1:
            self.b = 2e-5
        else:
            self.b = 2e-4

    def make_system(self):
        def system_linear(x,t, params):
            m1,jm1,m1_ = params["m1"],params["jm1"],params["m1_"]


            mu = self.mu * (1 - x[3])
            Lr = x[2] * (1 - mu) * self.k
            T=x[0]+x[1]
            DlT1 = self.b * (T * Lr / (1 + self.b * self.h * T))* x[0]
            DlT2 = self.b * (T * Lr / (1 + self.b * self.h * T)) * x[1]
            DlT=DlT2+DlT1
            DtT=m1 *(1 - self.e ** (-x[4]))* x[0]

            dx1dt = self.relu((self.a - self.anti * self.at) * (1 - (x[0]+x[1]) / self.T_max) * x[0]) - DlT1- DtT
            dx2dt = self.relu((self.a - self.anti * self.at) * (1 - (x[0]+x[1]) / self.T_max) * x[1]) - DlT2

            dx3dt = self.relu((DlT *self.j/ (10 + DlT)+DtT*jm1 / (10 + DtT))* (1 - x[2] / self.L_max)) \
                    - self.ml * x[2] -m1_*(1 - self.e ** (-x[4]))* x[2]

            dx4dt = -0.035*x[3]
            dx5dt =-0.9*x[4]

            return torch.cat([dx1dt, dx2dt.unsqueeze(dim=0), dx3dt,dx4dt.unsqueeze(dim=0),dx5dt.unsqueeze(dim=0)] ,dim=0)

        m1 = torch.nn.Parameter(torch.tensor([0.1894]))
        jm1 = torch.nn.Parameter(torch.tensor([0.0119]))
        m1_ = torch.nn.Parameter(torch.tensor([0.0468]))


        params = {"m1": m1, "jm1": jm1,'m1_':m1_}

        system = partial(system_linear, params=params)

        return system, params

    def make_system_s(self,params):
        def system_linear(x,t, params):
            m1,jm1,m1_ = params["m1"],params["jm1"],params["m1_"]

            mu = self.mu * (1 - x[3])
            Lr = x[2] * (1 - mu) * self.k
            T = x[0] + x[1]
            DlT1 = self.b * (T * Lr / (1 + self.b * self.h * T)) * x[0]
            DlT2 = self.b * (T * Lr / (1 + self.b * self.h * T)) * x[1]
            DlT = DlT2 + DlT1
            DtT = m1 * (1 - self.e ** (-x[4])) * x[0]

            dx1dt = self.relu((self.a - self.anti * self.at) * (1 - (x[0] + x[1]) / self.T_max) * x[0]) - DlT1 - DtT
            dx2dt = self.relu((self.a - self.anti * self.at) * (1 - (x[0] + x[1]) / self.T_max) * x[1]) - DlT2

            dx3dt = self.relu((DlT * self.j / (10 + DlT) + DtT * jm1 / (10 + DtT)) * (1 - x[2] / self.L_max)) \
                    - self.ml * x[2] - m1_ * (1 - self.e ** (-x[4])) * x[2]

            dx4dt = -0.035 * x[3]
            dx5dt = -0.9 * x[4]

            return torch.cat([dx1dt, dx2dt.unsqueeze(dim=0), dx3dt,dx4dt.unsqueeze(dim=0),dx5dt.unsqueeze(dim=0)] ,dim=0)



        system = partial(system_linear, params=params)

        return system

