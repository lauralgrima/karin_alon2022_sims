# messing around with Karin & Alon (2022) equations 
# original paper: https://journals.plos.org/ploscompbiol/article/comments?id=10.1371/journal.pcbi.1010340
# their github (some of the code is from there): https://github.com/omerka-weizmann/reward_taxis



import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint



# parameter values (mostly from Table 1)
u         = [0.05, 0.15, 0.5] # reward magnitude
mu        = 6 # dopamine gain 
a         = 1 # scaling factor 
b         = 1 # constant (magnitude independent component of reward activation)
C         = 15 # baseline DA when logR = g = 0 (no reward, no GABA)
alpha     = 0.7 # effectiveness of GABA inhibition 
omega     = 15 # adaptation time of DA after change in R(t)
omega_d   = 50 # NOT SURE WHERE THIS COMES FROM?
d0        = 5 # baseline firing (adapted)
v0        = 1

R_u       = calc_R_u(u,a,b) # spatial input field per reward
deltad_u  = calc_deltad_u(u,mu) # change in dopamine from baseline

def get_init_val()

def calc_d(max_t,C,deltad_u,alpha,plot=True):
    
    for deltad in deltad_u:
        for t in np.arange(0,max_t,0.01):
        
            d = C + deltad - alpha*g
        
    
        # call g iteratively - feedback loop 

# odeint for solving differential equations  
        


def calc_R_u(u,a,b,plot=False):
    # calculate simple reward function with scaling 
    R_u = [a*ur + b for ur in u]
    if plot:
        plt.plot(u,R_u)
        plt.xlabel('reward')
        plt.ylabel('reward field')
    return(R_u)


def calc_deltad_u(u,mu,a,b,plot=False):
    # calculate response of DA to logarithm of reward (sublinear) 
    R_u = calc_R_u(u,a,b,plot=False)
    deltad_u = mu*np.log(R_u)
    if plot:
        plt.plot(u,deltad_u)
        plt.xlabel('reward')
        plt.ylabel('change in dopamine from baseline')
    return(deltad_u)
        

def calc_g(C,d0,deltad_u,alpha)        
    # calculate change in g for a single timepoint 
    g = (C-d0+deltad_u)/alpha # GABAergic output integrates dopaminergic activity
    return(g)
        
def calc_dif_g(omega,d,d0):
    # calculate change in g for a single timepoint 
    dif_g = omega(d/d0-1)

      #  dydt = [omega_d*(C_+mu_*np.log(R(t))  - d), 
       #         omega*(d/d0_-1), 
        #        ]



        
def calc_v(d0,v0,d,plot=True):
    v = v0*np.array([da/d0 for da in d])
    if plot:
        plt.plot(d,v)
        plt.xlabel('dopamine')
        plt.ylabel('speed')
    return(v)



# FROM THEIR GITHUB
    

def load_eqs(omega_d,omega):
    
    def eqs(y,t,R):
        d,g = y
        dydt = [omega_d*(C_+mu_*np.log(R(t)) - alpha_*g - d), 
                omega*(d/d0_-1), 
                ]
        return dydt
    return eqs



def get_y0(r0=1,max_t = 100):
    t = np.arange(1, max_t,0.01)
    sol = odeint(eqs, [1,1], t,args=(lambda t: r0,))
    return sol[-1]

def run(r,t,r0=1):
    return odeint(eqs, get_y0(r0), t,args=(r,),rtol=1e-12,atol=1e-12)








