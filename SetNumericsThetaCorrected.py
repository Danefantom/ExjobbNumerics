# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:39:43 2021

@author: Daniel H
"""

"""
Calculate numerical solution to SET model with Feedback using rescaled probabalist's
Hermite polynomials appraoch

"""




import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lg
import scipy.special as sp


   
# Number of basis functions N_max

N_max = 300
# System parameters

Gamma_L = 1
Gamma_R = 1

relGamma = 0.05;
Gamma_L = 2*relGamma
Gamma_R = relGamma

#DeltaMu/Kbt = eta
eta = 0
f_L = 1/(np.exp(eta/2)+1)
f_R = 1/(np.exp(-eta/2)+1)

#Detector parameters


gamma = 1 #Only remaining free parameter
Lambda = 1

#sigma = gamma/(8*Lambda)
sigma = 0.05

# System matrix f - row i and row i + N_max corresponds to multiplying expansion
# with H_i and component a_i and d_i respectivly

f = np.zeros((2*N_max,2*N_max))

def logDoublefac(i):
    
    if i == -1 or i == 0:
        return 0
    MaxIndex = int(np.ceil(i/2) - 1)
    logSum = 0
    for k in range(MaxIndex+1):
        logSum += np.log(i-2*k)
    return logSum

def logfac(i):
    
    if i == -1 or i == 0:
        return 0
    logSum = 0
    for k in range(i):
        logSum += np.log(k+1)
    return logSum
    
def Cnm(i,j):
    n = i
    m = j
    
    if n == m:
        return 1/2
    elif (n+m)%2 == 0:
        return 0
    else:
        if n%2 != 0:
            temp = n
            n = m
            m = temp
        
        fac_1 =((-1)**((n+m-1)/2))/((m-n)*np.sqrt(2*np.pi))
        m_logff = logDoublefac(m)
        n_logff = logDoublefac(n-1)
        m_logf = logfac(m)
        n_logf = logfac(n)
        
        log_fac = m_logff+n_logff - (1/2)*(m_logf + n_logf)
        fac_fac = np.exp(log_fac)
        return fac_1*fac_fac
        
        
        
        
            

# Build L matrix part

for i in range(N_max):
    for j in range(N_max):
        Cnm_ij = Cnm(i,j)
        leftT = -Gamma_L*f_L*Cnm_ij*((-1)**(i+j))-Gamma_R*f_R*Cnm_ij
        rightT = Gamma_L*(1-f_L)*Cnm_ij*((-1)**(i+j))+Gamma_R*(1-f_R)*Cnm_ij
        
        f[i][j] += leftT
        f[i][j+N_max] += rightT 
        f[i+N_max][j] += -1*leftT 
        f[i+N_max][j+N_max] += -1*rightT
    
#Add drift term
for i in range(N_max):
    f[i][i] += -i
    f[i+N_max][i+N_max] += -i
    
    if i < N_max-2:
        f[i+2][i] += -np.sqrt((i+1)*(i+2))
        f[i+2+N_max][i+N_max] += -np.sqrt((i+1)*(i+2))
    if i < N_max -1:
        f[i+1][i] += -np.sqrt((i+1)/sigma)
        f[i+1+N_max][i+N_max] += np.sqrt((i+1)/sigma)
    

#Add diffusion term

#factor = (gamma*gamma)/(8*Lambda*sigma*sigma) -- 

for i in range(N_max):
    if i < N_max-2:
        f[i+2][i] += np.sqrt((i+1)*(i+2))
        f[i+2+N_max][i+N_max] += np.sqrt((i+1)*(i+2))
    
comps = lg.null_space(f)

#Normalization

a_0 = comps[0]
d_0 = comps[N_max]
norm = a_0+d_0
comps = comps/norm


#Build solution

nmbr_points = 500
x = np.linspace(-4,4,nmbr_points)
out = np.linspace(0,0,nmbr_points)
SepTime = np.linspace(0,0,nmbr_points)
NormConstant = 1/(np.sqrt(2*np.math.pi*sigma))

Psi_0 = NormConstant*np.exp(-(x**2)/(2*sigma))
Psi_prev = 0
    
for i in range(nmbr_points):
        k_1 = (1+sp.erf(np.sqrt(1/(2*sigma))))/2
        b = ((Gamma_R*(1-f_R)*k_1 + Gamma_L*(1-f_L)*(1-k_1)))
        a = (Gamma_L*f_L*k_1+Gamma_R*f_R*(1-k_1))
        SepTime[i] += (b/(a+b))*(1/np.sqrt(2*np.pi*sigma))*np.exp((-(x[i]+1)**2)/(2*sigma))
        SepTime[i] += (a/(a+b))*(1/np.sqrt(2*np.pi*sigma))*np.exp((-(x[i]-1)**2)/(2*sigma)) 
        for k in range(N_max):
            if k == 0:
                out[i] += Psi_0[i]*(comps[k]+comps[k+N_max])
            elif k == 1:
                Psi_new = Psi_0[i]*x[i]/(np.sqrt(sigma*(k)))
                Psi_prev =Psi_0[i]
                
                out[i]+= Psi_new*(comps[k]+comps[k+N_max])
            
            else:
                Psi_newVal = (Psi_new*x[i]/(np.sqrt(sigma*(k)))) - np.sqrt((k-1)/(k))*Psi_prev 
                out[i]+= (comps[k]+comps[k+N_max])*Psi_newVal
                Psi_prev = Psi_new
                Psi_new = Psi_newVal
                
#Comparision with Separation of Timescales approximation

Error = np.sqrt((out-SepTime)**2)
L2_error = np.trapz(Error,x)
L2_error = np.round(L2_error,3)
plt.plot(x,out,label='Numerical solution')
#Check norm
print("Integral over P(D):")
print(np.trapz(SepTime,x))
plt.plot(x,SepTime,linestyle='dashed',color='darkorange',label='Sep. of TS')
plt.plot(0,0,color='white',label=r'$L_2$ ' + 'error: ' + str(L2_error))
plt.title('Detector distribution in Steady state : $\Gamma/\gamma = $' + str(relGamma))
plt.xlabel(r'$D$')
plt.ylabel(r'$P(D)$')
plt.legend(prop={"size":8})

plt.show()        
        
        
        
        
        
        
        
        
        
        
        
