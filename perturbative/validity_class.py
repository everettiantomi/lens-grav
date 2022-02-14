import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import fsolve

class sieplasma(object):
    def __init__(self, theta_E_g, eta, zl, c, Dl, Ds, Dls, psi0_plasma_num, theta_0_num, B, C, delta_rs, deltab_10, deltab_20):

        self.theta_E_g = theta_E_g
        self.eta = eta
        self.psi0_plasma_num = psi0_plasma_num
        self.theta_0_num = theta_0_num
        self.B = B
        self.C = C
        self.delta_rs = delta_rs
        self.deltab_10 = deltab_10
        self.deltab_20 = deltab_20
        
        def f(r):
            tmp_f = r - theta_E_g + C/r * (r/B/theta_0_num)**C * psi0_plasma_num * np.exp(-(r/B/theta_0_num)**C)
            return tmp_f
            
        zero = root_scalar(f, bracket=[theta_E_g*.1, theta_E_g*1.9], method='bisect')
        self.theta_E = zero.root
        self.r = zero.root
        r = self.r
        
        def g(phi_k):
            tmp_g =  theta_E_g*eta*np.sin(2.*phi_k)/np.sqrt(1.-eta*np.cos(2.*phi_k)) - deltab_10*np.sin(phi_k) + deltab_20*np.cos(phi_k)
            return tmp_g
        
        phi_arr = np.array([np.pi/4, 5/4*np.pi])
        zeros_g = fsolve(g, phi_arr)
        self.zeros_phi = zeros_g
        zeros_phi = self.zeros_phi
        
        
        tmp_psi = theta_E_g*r*np.sqrt(1.-eta*np.cos(2.*zeros_phi)) + \
                  psi0_plasma_num*np.exp(-(r/B/theta_0_num)**C) 
        self.psi = tmp_psi


        tmp_dpsi = theta_E_g*r*(np.sqrt( 1. - eta*np.cos(2*zeros_phi)) - 1)
        self.dpsi = tmp_dpsi


        tmp_psi0 = theta_E_g*r + psi0_plasma_num*np.exp(-(r/B/theta_0_num)**C)
        self.psi0 = tmp_psi0


        tmp_psi_plasma = psi0_plasma_num*np.exp(-(r/B/theta_0_num)**C)
        self.psi_plasma = tmp_psi_plasma

        tmp_ddpsi_dr = theta_E_g*(np.sqrt( 1. - eta*np.cos(2*zeros_phi)) - 1)
        self.ddpsi_dr = tmp_ddpsi_dr


        tmp_ddpsi_dphi = theta_E_g*r*eta*np.sin(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi))
        self.ddpsi_dphi = tmp_ddpsi_dphi
        
        tmp_d2dpsi_dphi2 = theta_E_g*r*eta*( 2*np.cos(2.*zeros_phi)/np.sqrt(1.-eta*np.cos(2.*zeros_phi)) - (1.-eta*np.cos(2.*zeros_phi))**(-3/2)*eta*np.sin(2*zeros_phi)**2)
        self.d2dpsi_dphi2 = tmp_d2dpsi_dphi2


        tmp_d2psi0 = self.psi_plasma * ( - C*(C-1)/r**2*(r/B/theta_0_num)**C + (C/r*(r/B/theta_0_num)**C)**2 )
        self.d2psi0_dr2 = tmp_d2psi0
        
        delta_r = 1/(1 - self.d2psi0_dr2 )*(self.ddpsi_dr + deltab_10*np.cos(zeros_phi) + deltab_20*np.sin(zeros_phi) )

        self.delta_r = delta_r

        R = np.abs((delta_r[0]*(1 - self.d2psi0_dr2 ) - self.ddpsi_dr[0] - 1/r*self.d2dpsi_dphi2[0])/(delta_r[1]*(1 - self.d2psi0_dr2 ) - self.ddpsi_dr[1] - 1/r*self.d2dpsi_dphi2[1]))
        self.R = R
        
        dt = np.abs( (1 + zl)/c * Dl*Ds/Dls * r*(np.cos(zeros_phi[1])*deltab_10 + np.sin(zeros_phi[1])*deltab_20 - np.cos(zeros_phi[0])*deltab_10 - np.sin(zeros_phi[0])*deltab_20 + 1/r*self.dpsi[1] - 1/r*self.dpsi[0] ))
        self.dt = dt
        
        
        
