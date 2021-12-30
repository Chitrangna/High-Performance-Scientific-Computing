import torch
import torch.nn as nn

dtype = torch.float
device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class mma_process(nn.Module):
    def __init__(self):
        super(mma_process, self).__init__()
        self.F = torch.tensor(1, device=device, dtype=dtype) # m3/h
        self.F_cw = torch.tensor(0.159, device=device, dtype=dtype) # m3/h

        self.f_dash = torch.tensor(0.58, device=device, dtype=dtype) #
        self.V = torch.tensor(0.1, device=device, dtype=dtype) # m3
        self.V_0 = torch.tensor(0.02, device=device, dtype=dtype) # m3
        self.T_w0 = torch.tensor(293.2, device=device, dtype=dtype) # K
        self.rho = torch.tensor(866, device=device, dtype=dtype) # kg m-3
        self.rho_w = torch.tensor(1000, device=device, dtype=dtype) # kg m-3
        self.C_min = torch.tensor(6.4678, device=device, dtype=dtype) # kg mol m-3
        self.C_I_in = torch.tensor(8, device=device, dtype=dtype) # kg mol m-3
        self.U = torch.tensor(720, device=device, dtype=dtype) # kJ / (h.K.m2)

        self.F_I = torch.tensor(0.0032, device=device, dtype=dtype) # m3/h
        self.A = torch.tensor(2, device=device, dtype=dtype) # m2
        self.T_s = torch.tensor(0.008, device=device, dtype=dtype) # h
        self.T_in = torch.tensor(350, device=device, dtype=dtype) # 350
        self.R = torch.tensor(8.314, device=device, dtype=dtype) # kJ / (kg mol K)
        self.M_m = torch.tensor(100.12, device=device, dtype=dtype) # kg/ kgmol
        self.C_pw = torch.tensor(4.2, device=device, dtype=dtype) # kJ / (kg. K)
        self.C_p = torch.tensor(2.0, device=device, dtype=dtype) # kJ / (kg. K)
        self.del_h = torch.tensor(-57800, device=device, dtype=dtype) # kJ / (kg mol)

        self.A_I = torch.tensor(3.792e18, device=device, dtype=dtype) # 1/h
        self.A_tc = torch.tensor(3.8223e10, device=device, dtype=dtype) # m3 /(kgmol.h)
        self.A_td = torch.tensor(3.1457e11, device=device, dtype=dtype) # m3 /(kgmol.h)
        self.A_fm = torch.tensor(1.0067e15, device=device, dtype=dtype) # m3 /(kgmol.h)
        self.A_p = torch.tensor(1.77e9, device=device, dtype=dtype) # m 3 /(kgmol.h)

        self.E_I = torch.tensor(1.2877e5, device=device, dtype=dtype) # kJ/kgmol
        self.E_tc = torch.tensor(2.9422e3, device=device, dtype=dtype) # kJ/kgmol
        self.E_td = torch.tensor(2.9422e3, device=device, dtype=dtype) # kJ/kgmol
        self.E_fm = torch.tensor(7.4478e4, device=device, dtype=dtype) # kJ/kgmol
        self.E_p = torch.tensor(1.8283e4, device=device, dtype=dtype) # kJ/kgmol
    
    
    def dyn(self, t, x):
        
        h = torch.as_tensor(x, dtype=dtype, device=device)
        hdot = torch.zeros_like(h, dtype=dtype, device=device)
        
        '''
        h[0] = x[0]
        h[1] = x[1]
        h[2] = x[2]
        h[3] = x[3]
        h[4] = x[4]
        h[5] = x[5]
        '''
        
        # Reqd calculated values
        k_p = self.kcalc(self.A_p, self.E_p, h[4])
        k_fm = self.kcalc(self.A_fm, self.E_fm, h[4])
        k_I = self.kcalc(self.A_I, self.E_I, h[4])
        k_td = self.kcalc(self.A_td, self.E_td, h[4])
        k_tc = self.kcalc(self.A_tc, self.E_tc, h[4])

        # P_0 = torch.sqrt(2 * self.f_dash * h[1] * k_I/((self.F + self.V*k_I)*(k_td + k_tc)));
        P_0 = torch.sqrt(2 * self.f_dash * h[1] * k_I/(k_td + k_tc))

        hdot[0] = self.F*(self.C_min - h[0])/self.V;
        hdot[0] = hdot[0] - (k_p + k_fm) * h[0] * P_0;

        hdot[1] = (self.F_I * self.C_I_in - self.F * h[1])/self.V;
        hdot[1] = hdot[1] - k_I * h[1];

        hdot[2] = (0.5*k_tc + k_td) * torch.pow(P_0,2);
        hdot[2] = hdot[2] + k_fm * h[0] * P_0;
        hdot[2] = hdot[2] - (self.F * h[2])/self.V;

        hdot[3] = self.M_m * (k_p + k_fm) * h[0]  * P_0;
        hdot[3] = hdot[3] - (self.F * h[3])/self.V;

        hdot[4] = self.F*(self.T_in - h[4])/self.V;
        hdot[4] = hdot[4] - self.del_h * k_p * h[0] * P_0/(self.rho * self.C_p);
        hdot[4] = hdot[4] - self.U * self.A * (h[4] - h[5])/(self.rho * self.C_p * self.V);

        hdot[5] = self.F_cw * (self.T_w0 - h[5])/self.V_0;
        hdot[5] = hdot[5] + self.U * self.A * (h[4] - h[5])/(self.rho_w * self.C_pw * self.V_0);
        
        return hdot
        
        
    def ss(self, x):
        return self.dyn(0, x).cpu()
    
    
    def kcalc(self, A, E, T):
        k = A * torch.exp(-E/(self.R * T))
        return k
