import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

class PD_Controller(object):
    def __init__(self, dt, coef, kv, Kp, Kd):
        self.kv = kv
        
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        
        self.error = coef[0][0]
        self.error_prev = self.error
        
        self.u = 0
        
    def ControllerInput(self, coef, Vx):
        self.error = coef[0][0]
        kappa = self.curvature(coef, 0)
        
        P_term = self.Kp*(self.error)
        D_term = self.Kd*(self.error - self.error_prev) / self.dt
        FeedFowrad_Term = self.kv*(Vx**2)*kappa
        
        self.u = P_term + D_term + FeedFowrad_Term
        self.error_prev = self.error
    
    def curvature(self, coeff, x):
        c0, c1, c2, c3 = coeff.flatten()
        
        dy  = c1 + 2*c2*(x) + 3*c3*(x**2)
        d2y = 2*c2 + 6*c3*(x)
        
        return d2y / (1.0 + (dy)**2)**1.5
    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0-2*np.cos(X_ref/10)
    num_degree = 3
    num_point = 5
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    controller = PD_Controller(step_time, polynomialfit.coeff, ego_vehicle.kv,
                               Kp = 3.0, Kd = 0.5)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        
        frameconverter.convert(Points_ref[:num_point], ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        
        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


