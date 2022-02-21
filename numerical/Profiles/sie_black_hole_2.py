__author__ = 'gcrisnejo,everettiantomi'


import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class SIEBH2(LensProfileBase):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'eta', 'theta_E_1', 'x1', 'y1', 'theta_E_2', 'x2', 'y2']
    #lower_limit_default = {'A': 0.0, 'B': 0.0, 'k':1.0}
    #upper_limit_default = {'A': 100, 'B': 100, 'k': 100}

    def __init__(self):
        self.r_min = 10**(-50)
        super(SIEBH2, self).__init__()
        # alpha = 4*const.G * (mass*const.M_sun)/const.c**2/(r*const.Mpc)

    def function(self, x, y, theta_E, eta, theta_E_1, x1, y1, theta_E_2, x2, y2 , center_x=0, center_y=0):
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
            
        r1 = np.sqrt((x_- x1)**2 + (y_- y1)**2)
        r2 = np.sqrt((x_- x2)**2 + (y_- y2)**2)
        
        phi = theta_E*r*np.sqrt(1.-eta*np.cos(2.*theta)) +  theta_E_1**2*np.log(r1) + theta_E_2**2*np.log(r2)
        return phi

    def derivatives(self, x, y, theta_E, eta, theta_E_1, x1, y1, theta_E_2, x2, y2, center_x=0, center_y=0):
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
        r1 = np.sqrt((x_- x1)**2 + (y_- y1)**2)
        dr1_dx = (x_-x1)/r1
        dr1_dy = (y_-y1)/r1
        r2 = np.sqrt((x_- x2)**2 + (y_- y2)**2)
        dr2_dx = (x_-x2)/r2
        dr2_dy = (y_-y2)/r2
        
        
        
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta))
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        dphi_dx = dphi_dr * dr_dx + dphi_dtheta * dtheta_dx + theta_E_1**2/r1*dr1_dx + theta_E_2**2/r2*dr2_dx
        dphi_dy = dphi_dr * dr_dy + dphi_dtheta * dtheta_dy + theta_E_1**2/r1*dr1_dy + theta_E_2**2/r2*dr2_dy
        
        return dphi_dx, dphi_dy

    def hessian(self, x, y, theta_E, eta, theta_E_1, x1, y1, theta_E_2, x2, y2, center_x=0, center_y=0):
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
        r1 = np.sqrt((x_- x1)**2 + (y_- y1)**2)
        dr1_dx = (x_-x1)/r1
        dr1_dy = (y_-y1)/r1
        r2 = np.sqrt((x_- x2)**2 + (y_- y2)**2)
        dr2_dx = (x_-x2)/r2
        dr2_dy = (y_-y2)/r2
        # derivadas primeras
       
        
        dphi_dr = theta_E*np.sqrt(1.-eta*np.cos(2.*theta))
        dphi_dtheta = theta_E*r*eta*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        
        # derivadas segundas
        
        dphi_dr_dr = 0
        dphi_dr_dtheta = eta*theta_E*np.sin(2.*theta)/np.sqrt(1.-eta*np.cos(2.*theta))
        dphi_dtheta_dtheta = -eta*r*theta_E*(eta*(3.+np.cos(4.*theta))-4.*np.cos(2.*theta) )/2./(1.-eta*np.cos(2.*theta))**(3/2) 
        dphi_dtheta_dr = dphi_dr_dtheta
        
        bh1_xx = theta_E_1**2*( -2/r1**3*(x-x1)*dr1_dx + 1/r1**2)
        bh1_yy = theta_E_1**2*( -2/r1**3*(y-y1)*dr1_dy + 1/r1**2)
        bh1_xy = theta_E_1**2*( -2/r1**3*(x-x1)*dr1_dy  )
        bh2_xx = theta_E_2**2*( -2/r2**3*(x-x2)*dr2_dx + 1/r2**2)
        bh2_yy = theta_E_2**2*( -2/r2**3*(y-y2)*dr2_dy + 1/r2**2)
        bh2_xy = theta_E_2**2*( -2/r2**3*(x-x2)*dr2_dy  )
        
        
        f_xx = dr_dx*( dphi_dr_dr*dr_dx+dphi_dr*dr_dx_dr+dphi_dr_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dx+dphi_dr*dr_dx_dtheta+dphi_dtheta_dtheta*dtheta_dx+dphi_dtheta*dtheta_dx_dtheta ) + bh1_xx + bh2_xx
        f_yy = dr_dy*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dy* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta ) + bh1_yy + bh2_yy
        f_xy = dr_dx*( dphi_dr_dr*dr_dy+dphi_dr*dr_dy_dr+dphi_dr_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dr ) \
				+ dtheta_dx* ( dphi_dtheta_dr*dr_dy+dphi_dr*dr_dy_dtheta+dphi_dtheta_dtheta*dtheta_dy+dphi_dtheta*dtheta_dy_dtheta ) + bh1_xy + + bh2_xy
        return f_xx, f_xy, f_xy, f_yy
        
        
        
        
