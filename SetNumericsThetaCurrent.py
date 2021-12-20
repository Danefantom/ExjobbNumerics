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

def SepTimeCurr(det_val,fl,fr):
    k_1 = (1+sp.erf(np.sqrt(4*det_val)))/2
    k_2 = 1-k_1
    topp = fl*k_1*(1-fr)*k_1-fr*k_2*(1-fl)*k_2
    bott = fl*k_1+fr*k_2+(1-fl)*k_2+(1-fr)*k_1
    return topp/bott

Num_datapoints = 50
lambda_per_gamma = np.linspace(0.001,1,Num_datapoints)
Gamma_per_gamma = [0.01,0.1,0.5,1]
NumericalCurrent = np.zeros((4,Num_datapoints))
AnalyticalCurrent = np.zeros(Num_datapoints)

for q in range(4):
    print("Now doing current for: " + str(Gamma_per_gamma[q]))
    for p in range(Num_datapoints):
        
        
    
        # Number of basis functions N_max
        
        N_max = 300  
        # System parameters
        
        Gamma_L = 1
        Gamma_R = 1
        
        relGamma = Gamma_per_gamma[q]
        Gamma_L = relGamma
        Gamma_R = relGamma
        
        #DeltaMu/Kbt = eta
        eta = 5
        f_L = 1/(np.exp(eta/2)+1)
        f_R = 1/(np.exp(-eta/2)+1)
        
        
        AnalyticalCurrent[p] = SepTimeCurr(lambda_per_gamma[p],f_L,f_R)
        
        
        #Detector parameters
        
        
        gamma = 1 #Only remaining free parameter
        Lambda = 1
        
        #sigma = gamma/(8*Lambda)
        sigma = 1/(8*lambda_per_gamma[p])
        
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
                
                fac_1 = ((-1)**((n+m-1)/2))/((m-n)*np.sqrt(2*np.pi))
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
        
        
        #Find current
        Curr = 0
        for i in range(N_max):
            Curr += Cnm(i, 0)*((1-f_R)*comps[N_max+i]-comps[i]*f_R)
        
        NumericalCurrent[q][p] = Curr
                    
#Plot resulting currents

plt.plot(lambda_per_gamma,NumericalCurrent[0],linestyle='solid',color='blue',label='$\Gamma/\gamma = $' + str(Gamma_per_gamma[0]))
plt.plot(lambda_per_gamma,NumericalCurrent[1],linestyle='solid',color='red',label='$\Gamma/\gamma = $' + str(Gamma_per_gamma[1]))
plt.plot(lambda_per_gamma,NumericalCurrent[2],linestyle='solid',color='green',label='$\Gamma/\gamma = $' + str(Gamma_per_gamma[2]))
plt.plot(lambda_per_gamma,NumericalCurrent[3],linestyle='solid',color='darkorange',label='$\Gamma/\gamma = $' + str(Gamma_per_gamma[3]))
plt.plot(lambda_per_gamma,AnalyticalCurrent,linestyle='dashed',color='black',label='Sep. of TS')
plt.title('Steady state current: $I$/$\Gamma$ for $\Delta\mu$ = ' + str(eta))
plt.xlabel(r'$\lambda/\gamma$')
plt.ylabel(r'$I$ / $\Gamma$')
plt.legend(prop={"size":8})
plt.show()
print("All done! ;)")
     
        
        
        
        
        
        
        
        
        
        
        
