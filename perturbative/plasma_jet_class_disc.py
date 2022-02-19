import numpy as np
from scipy.optimize import root_scalar

class sieplasmadisc(object):
    def __init__(self, theta_E_g, eta, phi, psi0_plasma_num, theta_0r, theta_0y, alpha, delta_rs, deltab_10, deltab_20):

        self.theta_E_g = theta_E_g
        self.eta = eta
        self.phi = phi
        self.psi0_plasma_num = psi0_plasma_num
        self.theta_0r = theta_0r
        self.theta_0y = theta_0y
        self.alpha = alpha
        self.delta_rs = delta_rs
        self.deltab_10 = deltab_10
        self.deltab_20 = deltab_20
        
        def f(r):
            tmp_f = r - theta_E_g + 2/r * (r/theta_0r)**2 * np.exp(-(r/theta_0r)**2) * psi0_plasma_num
            return tmp_f
            
        zero = root_scalar(f, bracket=[theta_E_g*.1, theta_E_g*1.9], method='bisect')
        self.theta_E = zero.root
        self.r = zero.root
        r = self.r
        
        c = - ( r**2*np.sin(phi)**2*np.sin(alpha)**2 + r**2*np.cos(phi)**2  )/theta_0r**2 - (r**2*np.sin(phi)**2*np.cos(alpha)**2)/theta_0y**2
        
        dc_dr = 2/r*c
        d2c_dr2 = 2/r**2*c
        dc_dphi = - ( r**2*2*np.sin(phi)*np.cos(phi)*np.sin(alpha)**2 - r**2*2*np.sin(phi)*np.cos(phi)  )/theta_0r**2 - (r**2*2*np.sin(phi)*np.cos(phi)*np.cos(alpha)**2)/theta_0y**2
        d2c_dphi2 = (np.cos(phi)**2 - np.sin(phi)**2) * (- ( r**2*2*np.sin(alpha)**2 - r**2*2  )/theta_0r**2 - (r**2*2*np.cos(alpha)**2)/theta_0y**2)
        
        d = r**2*np.sin(phi)**2*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        dd_dr = 2/r*d
        d2d_dr2 = 2/r**2*d
        dd_dphi = r**2*2*np.sin(phi)*np.cos(phi)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        d2d_dphi2 = r**2*2*(np.cos(phi)**2 - np.sin(phi)**2)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
	
        self.d2psi0_dr2 = psi0_plasma_num * ( -2/theta_0r**2*np.exp(-(r/theta_0r)**2 ) + (2/r)**2*(r/theta_0r)**4*np.exp(-(r/theta_0r)**2))

 
        self.ddpsi_dr = theta_E_g*(np.sqrt( 1. - eta*np.cos(2*phi)) - 1) + psi0_plasma_num*( np.exp(c + d) * (dc_dr + dd_dr) + 2/r * (r/theta_0r)**2 * np.exp(-(r/theta_0r)**2 ) )

        self.ddpsi_dphi = theta_E_g*r*eta*np.sin(2.*phi)/np.sqrt(1.-eta*np.cos(2.*phi))   +   psi0_plasma_num * np.exp(c+d) * (dc_dphi + dd_dphi)
        

        self.d2dpsi_dphi2 = theta_E_g*r*eta*( 2*np.cos(2.*phi)/np.sqrt(1.-eta*np.cos(2.*phi)) - (1.-eta*np.cos(2.*phi))**(-3/2)*eta*np.sin(2*phi)**2)  +  psi0_plasma_num * np.exp(c+d) * (  (dc_dphi + dd_dphi)**2 + d2c_dphi2 + d2d_dphi2 )
        
        Delta = delta_rs**2 - ( 1/r*self.ddpsi_dphi - deltab_10*np.sin(phi) + deltab_20*np.cos(phi) )**2

        delta_r_1 = 1/(1 - self.d2psi0_dr2 )*(self.ddpsi_dr + deltab_10*np.cos(phi) + deltab_20*np.sin(phi) + np.sqrt(Delta))
        delta_r_2 = 1/(1 - self.d2psi0_dr2 )*(self.ddpsi_dr + deltab_10*np.cos(phi) + deltab_20*np.sin(phi) - np.sqrt(Delta))

        self.delta_r_1 = delta_r_1
        self.delta_r_2 = delta_r_2

        tmp_delta_r_criticline =  1/(1 - self.d2psi0_dr2 )*( self.ddpsi_dr + 1/r*self.d2dpsi_dphi2 )
        self.delta_r_criticline = tmp_delta_r_criticline
        
        tmp_caustic_1 = 1/r*(self.d2dpsi_dphi2 * np.cos(phi) + self.ddpsi_dphi * np.sin(phi) )
        self.caustic_1 = tmp_caustic_1
        tmp_caustic_2 = 1/r*(self.d2dpsi_dphi2 * np.sin(phi) - self.ddpsi_dphi * np.cos(phi) )
        self.caustic_2 = tmp_caustic_2
        
                                  
                                  
                                  
        
        
        
        
        

