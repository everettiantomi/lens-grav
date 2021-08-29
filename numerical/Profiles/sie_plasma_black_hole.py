__author__ = 'gcrisnejo,everettiantomi'


import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class PlasmaExponentialSIEBH(LensProfileBase):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'eta', 'A', 'B', 'C' ,'psi0_plasma', 'theta_0']
    #lower_limit_default = {'A': 0.0, 'B': 0.0, 'k':1.0}
    #upper_limit_default = {'A': 100, 'B': 100, 'k': 100}

    def __init__(self):
        self.r_min = 10**(-50)
        super(PlasmaExponentialSIE, self).__init__()
        # alpha = 4*const.G * (mass*const.M_sun)/const.c**2/(r*const.Mpc)

    def function(self, x, y, theta_E, eta, A, B, C, psi0_plasma, theta_0, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
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
        phi = theta_E*r*np.sqrt(1.-eta*np.cos(2.*theta))+psi0_plasma*np.exp(-(r/B/theta_0)**C)
        return phi

    def derivatives(self, x, y, theta_E, eta, A, B, C, psi0_plasma, theta_0, center_x=0, center_y=0):
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
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta))-C/r*psi0_plasma*(r/B/theta_0)**C*np.exp(-(r/B/theta_0)**C)
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        dphi_dx = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx
        dphi_dy = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy
        
        return dphi_dx, dphi_dy

    def hessian(self, x, y, theta_E, eta, A, B, C, psi0_plasma, theta_0, center_x=0, center_y=0):
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
        
        potencial = psi0_plasma*np.exp(-(r/B/theta_0)**C)
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta))-C/r*psi0_plasma*(r/B/theta_0)**C*np.exp(-(r/B/theta_0)**C)
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        # derivadas segundas
        
        dphi_dr_dr = potencial * C/r**2 *(r/B/theta_0)**C *(1.+C*(  (r/B/theta_0)**C-1.)  )
        dphi_dr_dtheta = eta*theta_E*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        dphi_dtheta_dtheta = -eta*r*theta_E*(eta*(3.+np.cos(4.*theta))-4.*np.cos(2.*theta) )/2./(1.-eta*np.cos(2.*theta))**(3/2) 
        dphi_dtheta_dr = dphi_dr_dtheta
        
        f_xx = dr_dx*( dphi_dr_dr*dr_dx+dphi_dr*dr_dx_dr+dphi_dr_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dx+dphi_dr*dr_dx_dtheta+dphi_dtheta_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dtheta )
        f_yy = dr_dy*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dy* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta )
        f_xy = dr_dx*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta )
        return f_xx, f_xy, f_xy, f_yy
