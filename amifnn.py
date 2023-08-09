import numpy as np

def get_fnn(x, dmax, tau, Rtol=10, Atol=2):
    eps = np.finfo(float).eps
    fnn = np.zeros(dmax)
    dim = []
    
    sigma = np.std(x)
    
    xzc = x-np.mean(x)              # zero centered x
    
    xc  = delayTS(xzc,tau,dim=1)    # current x delayed 
    
    for d in range(1,dmax+1):
        #print('d=',d)
        dim.append(d)
        
        xn = delayTS(xzc,tau,d+1)
        xc = xc[:xn.shape[0],:]
        
        for j in range(xc.shape[0]):
            dist = np.sum((xc - xc[j])**2, axis=1)
            
            id_so = np.argsort(dist)
            
            for id_se in range(1,5):
                try:
                    id_ne = id_so[id_se]
                    dc_ne = dist[id_ne]       # current distance of nearest neighbor

                    dn_ne = dc_ne+ (xn[id_ne,-1]-xn[j,-1])**2 

                    Rc    = np.sqrt(dc_ne) + eps
                    Rn    = np.sqrt(dn_ne) + eps

                    if np.sqrt(dn_ne-dc_ne)/Rc > Rtol:
                        fnn[d-1] += 1
                    elif (Rn/sigma)>Atol:
                        fnn[d-1] += 1
                    break
                except:
                    print('exception occured for j=',j)
                        
        xc = xn    
    return dim, fnn

def delayTS(x,tau,dim):
    xd = []
    s  = 0
    for i in range(dim):
        s  = i*tau 
        e  = len(x) - (dim-1-i)*tau
        #print('s=',s)
        #print('e=',e)
        if i == 0:
            xd = x[s:e].reshape(-1,1)
        else:
            xd = np.hstack((xd,x[s:e].reshape(-1,1)))
    return xd

def get_ami(xg, tau_max):
    dim = 2
    ami = np.zeros(tau_max)
    eps  = np.finfo(float).eps
    
    x    = (xg-min(xg))
    x    = x*(1-eps)/max(x)
    
    n_bins = np.array(np.ceil(np.log2(len(x))), dtype=int)#//5
    #print(n_bins)
    
    x    = np.array(np.floor(x*n_bins), dtype=int)
    
    for tau in range(tau_max):
        pxy      = np.zeros((n_bins,n_bins))
        #print(pxy)
        
        xd       = delayTS(x,tau,dim)
        #print(xd)
        
        for xt in xd:
            pxy[xt[0], xt[1]] +=1 
        
        pxy = pxy/xd.shape[0] + eps
        
        px  = np.sum(pxy, axis = 1)
        py  = np.sum(pxy, axis = 0)
        
        pd  = np.outer(px,py)
        temp     = pxy/pd
        temp     = pxy*np.log2(temp)
        ami[tau] = np.sum(temp.reshape(-1))
    return ami

def get_local_min(x, win=2): 
    id_min = []
    for i in range(len(x)):
        id_start = max(0, i-win)
        id_end = min(len(x), i+win+1)
        if x[i]==min(x[id_start:id_end]):
            id_min.append(i)
    return id_min[0]

def ami_fnn(x, sampling_time, tau_max, fnn_threshold, win =2, dim_max = 15):
    t = np.arange(len(x))*sampling_time

    ami = get_ami(x, tau_max)
    tau_ami = get_local_min(ami, win)
    dim, fnn = get_fnn(x, dim_max, tau_ami)

    fnn_zero = np.where(fnn<fnn_threshold)[0][0]

    dim_sel = dim[fnn_zero]
    delay_vec_sel = np.arange(dim_sel)*t[tau_ami]
    return ami, tau_ami, dim, fnn, fnn_zero, dim_sel, delay_vec_sel