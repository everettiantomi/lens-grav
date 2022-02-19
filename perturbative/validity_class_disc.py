import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import fsolve

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
        
        def g(phi_k):
            c = - ( r**2*np.sin(phi_k)**2*np.sin(alpha)**2 + r**2*np.cos(phi_k)**2  )/theta_0r**2 - (r**2*np.sin(phi_k)**2*np.cos(alpha)**2)/theta_0y**2
            dc_dphi = - ( r**2*2*np.sin(phi_k)*np.cos(phi_k)*np.sin(alpha)**2 - r**2*2*np.sin(phi_k)*np.cos(phi_k)  )/theta_0r**2 - (r**2*2*np.sin(phi_k)*np.cos(phi_k)*np.cos(alpha)**2)/theta_0y**2
            d = r**2*np.sin(phi_k)**2*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
            dd_dphi = r**2*2*np.sin(phi_k)*np.cos(phi_k)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
            tmp_g =  theta_E_g*eta*np.sin(2.*phi_k)/np.sqrt(1.-eta*np.cos(2.*phi_k)) + 1/r*psi0_plasma_num * np.exp(c+d) * (dc_dphi + dd_dphi) - deltab_10*np.sin(phi_k) + deltab_20*np.cos(phi_k)
            return tmp_g
        
        phi_arr = np.array([np.pi/4, 5/4*np.pi])
        zeros_g = fsolve(g, phi_arr)
        self.zeros_phi = zeros_g
        zeros_phi = self.zeros_phi
        
        
        c = - ( r**2*np.sin(zeros_phi)**2*np.sin(alpha)**2 + r**2*np.cos(zeros_phi)**2  )/theta_0r**2 - (r**2*np.sin(zeros_phi)**2*np.cos(alpha)**2)/theta_0y**2
        dc_dr = 2/r*c
        d2c_dr2 = 2/r**2*c
        dc_dphi = - ( r**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.sin(alpha)**2 - r**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)  )/theta_0r**2 - (r**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.cos(alpha)**2)/theta_0y**2
        d2c_dphi2 = (np.cos(zeros_phi)**2 - np.sin(zeros_phi)**2) * (- ( r**2*2*np.sin(alpha)**2 - r**2*2  )/theta_0r**2 - (r**2*2*np.cos(alpha)**2)/theta_0y**2)
        
        
        d = r**2*np.sin(zeros_phi)**2*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        dd_dr = 2/r*d
        d2d_dr2 = 2/r**2*d
        dd_dphi = r**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        d2d_dphi2 = r**2*2*(np.cos(zeros_phi)**2 - np.sin(zeros_phi)**2)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
	
        self.d2psi0_dr2 = psi0_plasma_num * ( -2/theta_0r**2*np.exp(-(r/theta_0r)**2 ) + (2/r)**2*(r/theta_0r)**4*np.exp(-(r/theta_0r)**2))

 
        self.ddpsi_dr = theta_E_g*(np.sqrt( 1. - eta*np.cos(2*zeros_phi)) - 1) + psi0_plasma_num*( np.exp(c + d) * (dc_dr + dd_dr) + 2/r * (r/theta_0r)**2 * np.exp(-(r/theta_0r)**2 ) )

        self.ddpsi_dphi = theta_E_g*r*eta*np.sin(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi))   +   psi0_plasma_num * np.exp(c+d) * (dc_dphi + dd_dphi)
        

        self.d2dpsi_dphi2 = theta_E_g*r*eta*( 2*np.cos(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi)) - (1.-eta*np.cos(2.*zeros_phi))**(-3/2)*eta*np.sin(2*zeros_phi)**2)  +  psi0_plasma_num * np.exp(c+d) * (  (dc_dphi + dd_dphi)**2 + d2c_dphi2 + d2d_dphi2 )
        
        delta_r = 1/(1 - self.d2psi0_dr2 )*(self.ddpsi_dr + deltab_10*np.cos(zeros_phi) + deltab_20*np.sin(zeros_phi) )

        self.delta_r = delta_r
        
        
        r_ = r + delta_r
        
        
        c = - ( r_**2*np.sin(zeros_phi)**2*np.sin(alpha)**2 + r_**2*np.cos(zer_os_phi)**2  )/theta_0r**2 - (r_**2*np.sin(zeros_phi)**2*np.cos(alpha)**2)/theta_0y**2
        dc_dr = 2/r_*c
        d2c_dr2 = 2/r_**2*c
        dc_dphi = - ( r_**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.sin(alpha)**2 - r_**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)  )/theta_0r**2 - (r_**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.cos(alpha)**2)/theta_0y**2
        d2c_dphi2 = (np.cos(zeros_phi)**2 - np.sin(zeros_phi)**2) * (- ( r_**2*2*np.sin(alpha)**2 - r_**2*2  )/theta_0r**2 - (r_**2*2*np.cos(alpha)**2)/theta_0y**2)
        d2c_dphidr = 2/r_*dc_dphi
        
        d = r_**2*np.sin(zeros_phi)**2*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        dd_dr = 2/r_*d
        d2d_dr2 = 2/r_**2*d
        dd_dphi = r_**2*2*np.sin(zeros_phi)*np.cos(zeros_phi)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        d2d_dphi2 = r_**2*2*(np.cos(zeros_phi)**2 - np.sin(zeros_phi)**2)*np.cos(alpha)**2*np.sin(alpha)**2 / (np.cos(alpha)**2/theta_0r**2 + np.sin(alpha)**2/theta_0y**2) * (1 / theta_0y**2 - 1 / theta_0r**2 )**2
        d2d_dphidr = 2/r_*dd_dphi
	
	
	self.dpsi0_dr = theta_E_g - psi0_plasma_num * 2/r_ * (r_/theta_0r)**2 * np.exp(-(r_/theta_0r)**2) 
	
        self.d2psi0_dr2 = psi0_plasma_num * ( -2/theta_0r**2*np.exp(-(r_/theta_0r)**2 ) + (2/r_)**2*(r_/theta_0r)**4*np.exp(-(r_/theta_0r)**2))

 
        self.ddpsi_dr = theta_E_g*(np.sqrt( 1. - eta*np.cos(2*zeros_phi)) - 1) + psi0_plasma_num*( np.exp(c + d) * (dc_dr + dd_dr) + 2/r_ * (r_/theta_0r)**2 * np.exp(-(r_/theta_0r)**2 ) )
        
        self.d2dpsi_dr2 = psi0_plasma_num*( np.exp(c + d)*( (dc_dr + dd_dr)**2 + d2c_dr2 + d2d_dr2 ) + np.exp(-(r_/theta_0r)**2) * ( 2/theta_0r**2 - (2/r_)**2*(r_/theta_0r)**4 ) )
        
        self.ddpsi_dphi = theta_E_g*r_*eta*np.sin(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi))   +   psi0_plasma_num * np.exp(c+d) * (dc_dphi + dd_dphi)
        

        self.d2dpsi_dphi2 = theta_E_g*r_*eta*( 2*np.cos(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi)) - (1.-eta*np.cos(2.*zeros_phi))**(-3/2)*eta*np.sin(2*zeros_phi)**2)  +  psi0_plasma_num * np.exp(c+d) * (  (dc_dphi + dd_dphi)**2 + d2c_dphi2 + d2d_dphi2 )
        
        self.d2dpsi_dphidr = theta_E_g*eta*np.sin(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi)) +  psi0_plasma_num * np.exp(c+d) * (  (dc_dphi + dd_dphi)*(dc_dr + dd_dr) + d2c_dphidr + d2d_dphidr ) 
        
        
        
        
        psi = theta_E_g*r_*np.sqrt(1.-eta*np.cos(2.*zeros_phi)) + psi0_plasma_num * np.exp(c + d) 
        dpsi_dr =  self.dpsi0_dr + self.ddpsi_dr
        d2psi_dr2 = self.d2psi0_dr2 + self.d2dpsi_dr2
        dpsi_dphi = self.ddpsi_dphi
        d2psi_dphi2 = self.d2dpsi_dphi2
        d2psi_dphidr = self.d2dpsi_dphidr
        
        
        mu = r_*( (1 - d2psi_dr2)*(r_ - dpsi_dr - 1/r_*d2psi_dphi2) - 1/r_*(1/r_*dpsi_dphi - d2psi_dphidr )**2  )**(-1)

        R = np.abs(mu[0]/mu[1])
        self.R = R
        
        t = (1 + zl)/c*Dl*Ds/Dls*(1/2* ( (r_*np.cos(zeros_phi) - deltab_10)**2 + (r_*np.sin(zeros_phi) - deltab_20)**2  ) - psi )
        
        self.t = t/24/60/60*0.000004848136811095**2
        
        dt = np.abs(t[0]-t[1])/24/60/60*0.000004848136811095**2 #convert seconds to days and arcsec^2 to rad
        self.dt = dt
        
        
        
