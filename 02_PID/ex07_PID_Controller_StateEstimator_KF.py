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
        
class KalmanFilter:
    def __init__(self, plantModel_d, x_0, Q_x = 0.01, Q_v = 0.02, R = 2.0, P_0 = 1.0):
        # Code
        self.A = plantModel_d.A
        self.B = plantModel_d.B
        self.C = plantModel_d.C
        
        self.x_estimate = np.array([[x_0],
                                    [0.0]])
        self.Q = np.array([[Q_x, 0.0],
                           [0.0, Q_v]])
        self.R = np.array([R])
        self.P_estimate = np.eye(2) * P_0
        
    def estimate(self, y_measure, u):
        # prediction
        self.x_predict = self.A @ self.x_estimate + self.B @ np.array([[u]])
        self.y_predict = self.C @ self.x_predict
        self.P_predict = self.A @ self.P_estimate @ self.A.T + self.Q
        
        # correction
        K = self.P_predict @ self.C.T @ np.linalg.inv(self.C @ self.P_predict @ self.C.T + self.R)
        self.x_estimate = self.x_predict + K @ (y_measure - self.y_predict)
        I = np.eye(self.A.shape[0])
        self.P_estimate = (I - K @ self.C) @ self.P_predict
        
        
        
if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    
    estimator = KalmanFilter(plant, plant.y_measure[0][0], Q_x = 0.001, Q_v = 0.01, R = 1.0, P_0 = 1.0)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time,
                                Kp = 1.8, Ki = 0.02, Kd = 3.0)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimator.estimate(plant.y_measure[0][0],controller.u)
        estimated_y.append(estimator.x_estimate[0][0])
        controller.ControllerInput(target_y, estimator.x_estimate[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r:',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
