import numpy as np

class sieplasmajet(object):
    def __init__(self, theta_E, r, eta, phi, psi0_plasma_num, theta_0_num, B, C, delta_rs, deltab_10, deltab_20):

        
        self.theta_E = theta_E
        self.r = r
        self.eta = eta
        self.phi = phi
        self.psi0_plasma_num = psi0_plasma_num
        self.theta_0_num = theta_0_num
        self.B = B
        self.C = C
        self.delta_rs = delta_rs
        self.deltab_10 = deltab_10
        self.deltab_20 = deltab_20
        
    def psi_func(self):
        tmp_psi = self.theta_E*self.r*np.sqrt(1.-self.eta*np.cos(2.*self.phi)) + \
                  self.psi0_plasma_num*np.exp(-(self.r/self.B/self.theta_0_num)**self.C) 
        self.psi = tmp_psi
    
    def dpsi_func(self):
        tmp_dpsi = self.theta_E*self.r*(np.sqrt( 1. - self.eta*np.cos(2*self.phi)) - 1)
        self.dpsi = tmp_dpsi
    
    def psi0_func(self):
        tmp_psi0 = self.theta_E*self.r + self.psi0_plasma_num*np.exp(-(self.r/self.B/self.theta_0_num)**self.C)
        self.psi0 = tmp_psi0
        
    def psi_plasma_func(self):
        tmp_psi_plasma = self.psi0_plasma_num*np.exp(-(self.r/self.B/self.theta_0_num)**self.C)
        self.psi_plasma = tmp_psi_plasma
        
    def ddpsi_dr_func(self):
        tmp_ddpsi = self.theta_E*(np.sqrt( 1. - self.eta*np.cos(2*self.phi)) - 1)
        self.ddpsi_dr = tmp_ddpsi
    
    def ddpsi_dphi_func(self):
        tmp_ddpsi = self.theta_E*self.r*self.eta*np.sin(2.*self.phi)/np.sqrt(1.-self.eta*np.cos(2.*self.phi))
        self.ddpsi_dphi = tmp_ddpsi
    
    def d2psi0_dr2_func(self):

        
        self.psi_plasma_func()
        tmp_d2psi0 = self.psi_plasma * self.C/self.r**2 *(self.r/self.B/self.theta_0_num)**self.C *(1.+self.C*( (self.r/self.B/self.theta_0_num)**(self.C-1)))
        self.d2psi0_dr2 = tmp_d2psi0

    def delta_r(self):

        self.ddpsi_dr_func()
        self.ddpsi_dphi_func()
        self.d2psi0_dr2_func()
        
        ddpsi_dphi = self.ddpsi_dphi
        d2psi0_dr2 = self.d2psi0_dr2
        ddpsi_dr = self.ddpsi_dr
        delta_rs = self.delta_rs
        deltab_10 = self.deltab_10
        deltab_20 = self.deltab_20
        
        Delta = delta_rs**2 - ( 1/self.r*ddpsi_dphi - deltab_10*np.sin(self.phi) + deltab_20*np.cos(self.phi) )**2
        
        delta_r_1 = 1/(1 - d2psi0_dr2 )*(ddpsi_dr + deltab_10*np.cos(self.phi) + deltab_20*np.sin(self.phi) + np.sqrt(Delta))
        delta_r_2 = 1/(1 - d2psi0_dr2 )*(ddpsi_dr + deltab_10*np.cos(self.phi) + deltab_20*np.sin(self.phi) - np.sqrt(Delta))

        self.delta_r_1 = delta_r_1
        self.delta_r_2 = delta_r_2