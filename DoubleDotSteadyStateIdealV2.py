# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:08:38 2021

@author: Daniel Holst
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:48:02 2021

@author: Daniel Holst
"""

import numpy as np
import scipy.linalg as ln
import scipy.special as sp
import matplotlib.pyplot as plt

num_points = 100
    
#System parameters
Gamma_L = 100
Gamma_R = 100

delta_mu = 0
sepFac = 10

mu_l = -delta_mu/2
mu_r = delta_mu/2

e_0 = 0
e_l = -sepFac
e_u = sepFac

g = 100

def f_l(x):
    outval = 1/(np.exp(x-mu_l)+1)
    return outval

def f_r(x):
    outval = 1/(np.exp(x-mu_r)+1)
    return outval

f_l0 = f_l(e_0)
f_ll = 1 #f_l(e_l)
f_lu = 0 #f_l(e_u)

f_r0 = f_r(e_0)
f_rl = 1 #f_r(e_l)
f_ru = 0 #f_r(e_u)

delta_e0 = e_0 - e_u
delta_eL = 0
delta_eR = e_u - e_0


#Detector parameters

gamma_2 = 100
lambda_2 = 1


sigma_2 = gamma_2/(8*lambda_2)

#Numerics parameters

n_poly = 300

#First n_poly rows for a, next for b and so on... 3npoly Re{d}, 4npoly Im{d}

system_matrix = np.zeros((5*n_poly,5*n_poly))


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
    
def I_nm(i,j):
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
        
        
#Build L_0

for i in range(n_poly):
    
    system_matrix[i][i] += (-Gamma_L*f_l0)  
    system_matrix[i+n_poly][i] += Gamma_L*f_l0
    
    

#Build L_R

for i in range(n_poly):
    for j in range(n_poly):
            
        I_val = I_nm(i,j)

        system_matrix[i][j+n_poly] += Gamma_L*I_val 
        system_matrix[i][j+2*n_poly] += Gamma_R*(1-f_r0)*I_val 
    
        system_matrix[i+n_poly][j+n_poly] += -Gamma_L*I_val
        system_matrix[i+n_poly][j+4*n_poly] += -2*g*I_val
        
        
        system_matrix[i+2*n_poly][j+2*n_poly] += -Gamma_R*(1-f_r0)*I_val
        system_matrix[i+2*n_poly][j+4*n_poly] += 2*g*I_val
        
        
        system_matrix[i+3*n_poly][j+3*n_poly] += -(((Gamma_L*(1-f_lu)+Gamma_R*(1-f_r0))/2)+2*lambda_2)*I_val
        system_matrix[i+3*n_poly][j+4*n_poly] += delta_eR*I_val
        
        system_matrix[i+4*n_poly][j+n_poly] += g*I_val
        system_matrix[i+4*n_poly][j+2*n_poly] += -g*I_val
        system_matrix[i+4*n_poly][j+3*n_poly] += -delta_eR*I_val
        system_matrix[i+4*n_poly][j+4*n_poly] += -(((Gamma_L*(1-f_lu)+Gamma_R*(1-f_r0)))/2 + 2*lambda_2)*I_val
     
#Build L_L

for i in range(n_poly):
    for j in range(n_poly):
            
        I_val = I_nm(i,j)*((-1)**(i+j))

        system_matrix[i+n_poly][j+4*n_poly] += -2*g*I_val
        
        system_matrix[i+2*n_poly][j+4*n_poly] += 2*g*I_val
        
        system_matrix[i+3*n_poly][j+3*n_poly] += -(2*lambda_2)*I_val
        
        system_matrix[i+4*n_poly][j+n_poly] += g*I_val
        system_matrix[i+4*n_poly][j+2*n_poly] += -g*I_val
        system_matrix[i+4*n_poly][j+4*n_poly] += -(2*lambda_2)*I_val
        

#Build back-action term

for i in range(n_poly):
    system_matrix[i+3*n_poly][i+3*n_poly] += -2*lambda_2
    system_matrix[i+4*n_poly][i+4*n_poly] += -2*lambda_2
    
#Build drift term

for i in range(n_poly):
    system_matrix[i][i] += gamma_2
    system_matrix[i+n_poly][i+n_poly] += gamma_2
    system_matrix[i+2*n_poly][i+2*n_poly] += gamma_2
    system_matrix[i+3*n_poly][i+3*n_poly] += gamma_2
    system_matrix[i+4*n_poly][i+4*n_poly] += gamma_2
    
        
        
for i in range(n_poly):
    system_matrix[i][i] += -gamma_2*(i+1)
    system_matrix[i+n_poly][i+n_poly] += -gamma_2*(i+1)
    system_matrix[i+2*n_poly][i+2*n_poly] += -gamma_2*(i+1)

    system_matrix[i+3*n_poly][i+3*n_poly] += -gamma_2*(i+1)
    system_matrix[i+4*n_poly][i+4*n_poly] += -gamma_2*(i+1)
    
    
    if i < n_poly - 2:   
        system_matrix[i+2][i] += -gamma_2*np.sqrt((i+1)*(i+2))
        system_matrix[i+n_poly+2][i+n_poly] += -gamma_2*np.sqrt((i+1)*(i+2))
        system_matrix[i+2*n_poly+2][i+2*n_poly] += -gamma_2*np.sqrt((i+1)*(i+2))
        
        system_matrix[i+3*n_poly+2][i+3*n_poly] += -gamma_2*np.sqrt((i+1)*(i+2))
        system_matrix[i+4*n_poly+2][i+4*n_poly] += -gamma_2*np.sqrt((i+1)*(i+2))
            
    
    if i < n_poly -1:
        system_matrix[i+n_poly+1][i+n_poly] += -gamma_2*np.sqrt((i+1)/sigma_2)
        system_matrix[i+2*n_poly+1][i+2*n_poly] += gamma_2*np.sqrt((i+1)/sigma_2)

#Build diffusion term
for i in range(n_poly-2):
    system_matrix[i+2][i] += gamma_2*np.sqrt((i+1)*(i+2))
    system_matrix[i+n_poly+2][i+n_poly] += gamma_2*np.sqrt((i+1)*(i+2))
    system_matrix[i+2*n_poly+2][i+2*n_poly] += gamma_2*np.sqrt((i+1)*(i+2))
    system_matrix[i+3*n_poly+2][i+3*n_poly] += gamma_2*np.sqrt((i+1)*(i+2))
    system_matrix[i+4*n_poly+2][i+4*n_poly] += gamma_2*np.sqrt((i+1)*(i+2))
        


res = ln.null_space(system_matrix)
a_0 = res[0]
b_0 = res[n_poly]
c_0 = res[2*n_poly]

norm = a_0 + b_0 + c_0

res = res/norm
 
#Build solution

nmbr_points = 500
x = np.linspace(-10,10,nmbr_points)
out = np.linspace(0,0,nmbr_points)
NormConstant = 1/(np.sqrt(2*np.math.pi*sigma_2))

Psi_0 = NormConstant*np.exp(-(x**2)/(2*sigma_2))
Psi_prev = 0
    
for i in range(nmbr_points):
        for k in range(n_poly):
            
            comp = res[0+k] + res[k+n_poly] + res[2*n_poly+k] 
            if k == 0:
                out[i] += Psi_0[i]*comp
            elif k == 1:
                Psi_new = Psi_0[i]*x[i]/(np.sqrt(sigma_2*(k)))
                Psi_prev =Psi_0[i]
                
                out[i]+= Psi_new*comp
            
            else:
                Psi_newVal = (Psi_new*x[i]/(np.sqrt(sigma_2*(k)))) - np.sqrt((k-1)/(k))*Psi_prev 
                out[i]+= comp*Psi_newVal
                Psi_prev = Psi_new
                Psi_new = Psi_newVal
                
plt.plot(x,out)

sep_sol = np.linspace(0,0,nmbr_points)
k_2 = 0.5*(1-sp.erf(np.sqrt(4*lambda_2/gamma_2)))

for i in range(nmbr_points):
    g_1 = f_l0*Gamma_L
    g_2 = 0
    g_3 = k_2*Gamma_L
    g_4 = (1-f_r0)*(1-k_2)*Gamma_R
    g_5 = 0.5*Gamma_L
    g_6 = (1-f_r0)*0.5*Gamma_R
    
    alpha = e_u*0.5
    w = (g_5+g_6)/2 + 2*lambda_2
    
    Omega = (alpha**2+w**2)*(g_3*g_4+g_1*g_4+g_2*g_3) + 2*g**2*w*(2*(g_1+g_2)+g_3+g_4)
    rho_ll = (1/Omega)*(g_1*g_4*(alpha**2+w**2) + 2*g**2*w*(g_1+g_2))
    rho_00 = (1/Omega)*(g_3*g_4*(alpha**2+w**2) + 2*g**2*w*(g_3+g_4))
    rho_rr = (1/Omega)*(g_2*g_3*(alpha**2+w**2) + 2*g**2*w*(g_1+g_2))
    
    c = -4*lambda_2/gamma_2
    sep_sol[i] = np.sqrt(4*lambda_2/(np.pi*gamma_2))*(rho_00*np.exp(c*(x[i]**2))+rho_ll*np.exp(c*((x[i]+1)**2)) + rho_rr*np.exp(c*((x[i]-1)**2)))
    
Error = np.sqrt((out-sep_sol)**2)
L2_error = np.trapz(Error,x)
L2_error = np.round(L2_error,3)
plt.plot(x,out,linestyle='solid',color='black',label='Numerical solution')
#Check norm
print("Integral over P(D):")
print(np.trapz(out,x))
plt.plot(x,sep_sol,linestyle='dashed',color='darkorange',label='Sep. of TS')
plt.plot(0,0,color='white',label=r'$L_2$ ' + 'error: ' + str(L2_error))
plt.title('Detector distribution in Steady state : $\Gamma/\gamma = $' + str(Gamma_L/gamma_2) + ' $\lambda = $' + str(lambda_2))
plt.xlabel(r'$D$')
plt.ylabel(r'$P(D)$')
plt.legend(prop={"size":8})

plt.show()












