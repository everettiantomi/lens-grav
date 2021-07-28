__author__ = 'gcrisnejo,everettiantomi'


import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class PlasmaGaussianSIEcantorot(LensProfileBase):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'eta', 'a', 'b1', 'b2', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'd4', 'e1', 'e2', 'e3', 'e4', 'e5', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'psi0_plasma']
    #lower_limit_default = {'A': 0.0, 'B': 0.0, 'k':1.0}
    #upper_limit_default = {'A': 100, 'B': 100, 'k': 100}

    def __init__(self):
        self.r_min = 10**(-50)
        super(PlasmaGaussianSIEcantorot, self).__init__()
        # alpha = 4*const.G * (mass*const.M_sun)/const.c**2/(r*const.Mpc)

    def function(self, x, y, theta_E, eta, a, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, f1, f2, f3, f4, f5, f6, psi0_plasma, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)º
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: lensing potential
        """

        
        x_ = x - center_x
        y_ = y - center_y
        a = np.sqrt(x_**2 + y_**2)
        theta = np.arctan2(y_,x_)
        
        if isinstance(a, int) or isinstance(a, float):
            r = max(self.r_min, a)
        else:
            r = np.empty_like(a)
            r[a > self.r_min] = a[a > self.r_min]  #in the SIS regime
            r[a <= self.r_min] = self.r_min
        
        N_e = (a + 
        b1*np.abs(x_) + b2*np.abs(y_) + 
        c1*x_**2 + c2*np.abs(x_)*np.abs(y_) + c3*y_**2 + 
        d1*np.abs(x_)**3 + d2*x_**2*np.abs(y_) + d3*np.abs(x_)*y_**2 + d4*np.abs(y_)**3 + 
        e1*x_**4 + e2*np.abs(x_)**3*np.abs(y_) + e3*x_**2*y_**2 + e4*np.abs(x_)*np.abs(y_)**3 + e5*y_**4 + 
        f1*np.abs(x_)**5 + f2*x_**4*np.abs(y_) + f3*np.abs(x_)**3*y_**2 + f4*x_**2*np.abs(y_)**3 + f5*np.abs(x_)*y_**4 + f6*np.abs(y_)**5)
        
        phi = theta_E*r*np.sqrt(1.-eta*np.cos(2.*theta))   +   psi0_plasma*N_e
        
        return phi

    def derivatives(self, x, y, theta_E, eta, a, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, f1, f2, f3, f4, f5, f6, psi0_plasma, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: deflection angle (in radian)
        """
        x_ = x - center_x
        y_ = y - center_y
        a = np.sqrt(x_**2 + y_**2)
        theta = np.arctan2(y_,x_)
        if isinstance(a, int) or isinstance(a, float):
            r = max(self.r_min, a)
        else:
            r = np.empty_like(a)
            r[a > self.r_min] = a[a > self.r_min]  #in the SIS regime
            r[a <= self.r_min] = self.r_min
        
        dr_dx = x_/r
        dr_dy = y_/r
        dtheta_dx = -y_/r**2
        dtheta_dy = x_/r**2
        
        # de canto |cos\phi| y r*sin\phi=sqrt( (r*sin\phi)^2+algo_chiquito^2 )
        
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta)) 
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        dphiplasma_dx = psi0_plasma*(b1*np.abs(x_)/x_  + 2*c1*x_ + c2*np.abs(x_)/x_*np.abs(y_)  + 3*d1*x_**2*np.abs(x_)/x_ + 2*d2*x_*np.abs(y_) + d3*np.abs(x_)/x_*y_**2 + 4*e1*x_**3 + 3*e2*x_**2*np.abs(x_)/x_*np.abs(y_) + 2*e3*x_*y_**2 + e4*np.abs(x_)/x_*np.abs(y_)**3 +  5*f1*x_**4*np.abs(x_)/x_ + 4*f2*x_**3*np.abs(y_) + 3*f3*x**2*np.abs(x_)/x_*y_**2 + 2*f4*x_*np.abs(y_)**3 + f5*np.abs(x_)/x_*y_**4)
        
        
        dphiplasma_dy = psi0_plasma*(b2*np.abs(y_)/y_  + c2*np.abs(x_)*np.abs(y_)/y_  + 2*c3*y_ + d2*x_**2*np.abs(y_)/y_  + 2*d3*np.abs(x_)*y_ + 3*d4*y**2*np.abs(y_)/y_  + e2*np.abs(x_)**3*np.abs(y_)/y_  + 2*e3*x_**2*y_ + 3*e4*np.abs(x_)*y_**2*np.abs(y_)/y_  + 4*e5*y_**3 + f2*x_**4*np.abs(y_)/y_  + 2*f3*np.abs(x_)**3*y_ + 3*f4*x_**2*y_**2*np.abs(y_)/y_  + 4*f5*np.abs(x_)*y_**3 + 5*f6*y_**4*np.abs(y_)/y_)
        
        
        
        dphi_dx = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx + dphiplasma_dx
        dphi_dy = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy + dphiplasma_dy
        
        return dphi_dx, dphi_dy

    def hessian(self, x, y, theta_E, eta, a, b1, b2, c1, c2, c3, d1, d2, d3, d4, e1, e2, e3, e4, e5, f1, f2, f3, f4, f5, f6, psi0_plasma, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: hessian matrix (in radian)
        """
        x_ = x - center_x
        y_ = y - center_y
        a = np.sqrt(x_**2 + y_**2)
        theta = np.arctan2(y_,x_)
        if isinstance(a, int) or isinstance(a, float):
            r = max(self.r_min, a)
        else:
            r = np.empty_like(a)
            r[a > self.r_min] = a[a > self.r_min]  #in the SIS regime
            r[a <= self.r_min] = self.r_min
        
        dr_dx = x_/r; dr_dy = y_/r; dtheta_dx = -y_/r**2; dtheta_dy = x_/r**2; dr_dx_dr = 0.0; dr_dy_dr = 0.0; dr_dx_dtheta = -np.sin(theta)
        dr_dy_dtheta = np.cos(theta); dtheta_dx_dr = np.sin(theta)/r**2; dtheta_dy_dr = -np.cos(theta)/r**2
        dtheta_dx_dtheta = -np.cos(theta)/r; dtheta_dy_dtheta = -np.sin(theta)/r
        
        # derivadas primeras
        #--------------------------
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta))
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        #dphi_dx = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx
        #dphi_dy = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy
        # ------------------------
        
        # derivadas segundas
        
        dphi_dr_dr =0.
        dphi_dr_dtheta = eta*theta_E*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta)) 
        
        dphi_dtheta_dtheta = -eta*r*theta_E*(eta*(3.+np.cos(4.*theta))-4.*np.cos(2.*theta) )/2./(1.-eta*np.cos(2.*theta))**(3/2) 
        
        dphi_dtheta_dr = dphi_dr_dtheta
        
        dphiplasma_dx_dx = 2*c1 + 6*d1*np.abs(x_) + 2*d2*np.abs(y_) + 12*e1*x_**2 + 6*e2*np.abs(x_)*np.abs(y_) + 2*e3*y_**2 + 20*f1*x_**2*np.abs(x_) + 12*f2*x_**2*np.abs(y_) + 6*f3*np.abs(x_)*y_**2 + 2*f4*np.abs(y_)**3 
        dphiplasma_dy_dy = 2*c3 + 2*d3*np.abs(x_) + 6*d4*np.abs(y_) + 2*e3*x_**2 + 6*e4*np.abs(x_)*np.abs(y_)  + 12*e5*y_**2 + 2*f3*np.abs(x_)**3 + 6*f4*x_**2*np.abs(y_) + 12*f5*np.abs(x_)*y_**2 + 20*f6*y_**2*np.abs(y_)
        dphiplasma_dx_dy = (c2*np.abs(x_)/x_*np.abs(y_)/y_ + 2*d2*x_*np.abs(y_)/y_ + 2*d3*np.abs(x_)/x_*y_ + 3*e2*x_**2*np.abs(x_)/x_*np.abs(y_)/y_ + 4*e3*x_*y_ + 3*e4*np.abs(x_)/x_*y_**2*np.abs(y_)/y_ + 4*f2*x_**3*np.abs(y_)/y_ + 6*f3*x_**2*np.abs(x_)/x_*y_ + 6*f4*x_*y_**2*np.abs(y_)/y_ + 4*f5*np.abs(x_)/x_*y_**3)
        
        
        f_xx = dr_dx*( dphi_dr_dr*dr_dx+dphi_dr*dr_dx_dr+dphi_dr_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dx+dphi_dr*dr_dx_dtheta+dphi_dtheta_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dtheta ) + dphiplasma_dx_dx
        f_yy = dr_dy*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dy* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta )+ dphiplasma_dy_dy
        f_xy = dr_dx*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta ) + dphiplasma_dx_dy
        return f_xx, f_xy, f_xy, f_yy
