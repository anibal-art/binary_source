import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
def A_u(u):
    A_t = (u**2 + 2)/(u*(u**2 + 4)**(.5))
    return A_t

def A_Binary_SC(t,t_0, u_0,t_E,xiE, omega, phi_0,lambda_xi, q , binary_flux_ratio,fs,fb):
    '''
    Somayeh and Carmen implementation
    t (1-d array): time
    t_0(float): time of closest approach
    phi_0 (float): phase
    q(float): mass_ratio
    P (float): period in days
    xiE(float): separation from the source to the barycenter in theta_E units
    binary_flux_ratio(float): binary flux ratio
    
    '''
    
    Omega =  omega*(t - t_0) + phi_0
    a1 =  q/(1+q)     # primary's orbital radius
    a2 =  -1/(1+q) 
    tau =  ((t-t_0)/t_E)
    
    delta_tau = xiE*(np.cos(Omega)-np.cos(phi_0))
    delta_beta = xiE*np.sin(lambda_xi)*(np.sin(Omega)-np.sin(phi_0))
    u1 = np.sqrt((u_0 + a1*delta_beta)**2 + (tau + a1*delta_tau)**2)
    
    m1 = q/(1+q)
    
    # This is the pyLIMA election:
    dx2 = -(1/(1+q))*np.cos(Omega) - (q/(1+q))*np.cos(phi_0)
    dy2 = -(1/(1+q))*np.sin(Omega) - (q/(1+q))*np.sin(phi_0)
    u2 = np.sqrt((u_0 + xiE*dy2)**2 + (tau + xiE*dx2)**2)
    # u2 = np.sqrt((u_0 + a2*delta_beta)**2 + (tau + a2*delta_tau)**2)  # This is the election of Miyazaki with symetry around the barycenter
    A1_binary = A_u(u1)
    A2_binary = A_u(u2)
    A_binary = A1_binary + binary_flux_ratio*A2_binary
    # F_binary1 = F(f_s, A_binary, fb)
    return A_binary 




case2_params = OrderedDict([('t0', 50.0),
             ('u0', 0.1),
             ('tE', 173.14568368055558),
             ('xi_para', 0.4),
             ('xi_perp', 0.0),
             ('xi_angular_velocity', 0.06124401127235416),
             ('xi_phase', 0.0),
             ('xi_inclination', 1.5707963267948966),
             ('xi_mass_ratio', 71.42857142857143),
             ('q_flux_G', 0.2),
             ('fsource_Simulation', 1),
             ('ftotal_Simulation', 0)])

t = np.linspace(-500, 500, 5000)
t_0 = case2_params['t0']
u_0= case2_params['u0']
t_E= case2_params['tE']
xiE= case2_params['xi_para']
omega= case2_params['xi_angular_velocity']
phi_0= case2_params['xi_phase']
lambda_xi= case2_params['xi_inclination']
q= case2_params['xi_mass_ratio']
binary_flux_ratio= case2_params['q_flux_G']
fs= case2_params['fsource_Simulation']
fb= case2_params['ftotal_Simulation']


A_binary = A_Binary_SC(t,t_0, u_0,t_E,xiE, omega, phi_0,lambda_xi, q , binary_flux_ratio,fs,fb)

plt.plot(t, np.log10(A_binary), linestyle='--', label='Somayeh - Carmen')
plt.xlabel("Time")
plt.ylabel(r"$LOG(A(t))$")
plt.legend()
plt.tight_layout()
plt.show()
