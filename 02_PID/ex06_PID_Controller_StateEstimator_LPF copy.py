from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, ref, measure, dt = 0.1, Kp = 0.1, Ki = 0.1, Kd = 0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        self.error = ref - measure
        self.error_prev = self.error
        self.error_sum = 0.0
        
        self.u = 0.0
    
    def ControllerInput(self, ref, measure):
        
        self.error = ref - measure
        self.error_sum += self.error * self.dt
        
        P_term = self.Kp * (self.error)
        I_term = self.Ki * (self.error_sum)
        D_term = self.Kd * (self.error - self.error_prev) / self.dt
        
        self.u = P_term + I_term + D_term
        self.error_prev = self.error
        
class LowPassFilter:
    def __init__(self, y_0, alpha = 0.7):
        self.y_estimate = y_0
        self.alpha = alpha
 
    def estimate(self, y_measure):
        self.y_estimate = (self.alpha) * self.y_estimate + (1 - self.alpha) * y_measure


if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    
    estimator = LowPassFilter(plant.y_measure[0][0], alpha = 0.8)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time,
                                Kp = 1.8, Ki = 0.02, Kd = 3.0)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimator.estimate(plant.y_measure[0][0])
        estimated_y.append(estimator.y_estimate)
        controller.ControllerInput(target_y, estimator.y_estimate)
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r-',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
