import torch

dtype = torch.float
device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def num_jacobian(func, Z0):
    nZ0 = len(Z0)
    JacobF = torch.zeros((nZ0,nZ0), dtype=dtype, device=device)
    
    for i in range(nZ0):
        eps = Z0[i]/10000
        if eps == 0:
            eps = 1 / 10000 
        
        Zp = Z0
        Zp[i] = Zp[i] + eps 
        fp = func(Zp)
        
        Zn = Z0
        Zn[i] = Zn[i] - eps 
        fn = func(Zn)
        
        JacobF[:,i] = (fp - fn) / (2 * eps)
    
    return JacobF


def radius(p):
    eigenvalues = torch.eig(p)[0]
    r = torch.max(torch.abs(eigenvalues))
    return r
