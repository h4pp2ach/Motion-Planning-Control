import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

class KalmanFilter:
    def __init__(self, y_0, dt = 0.1, m = 10, c = 2, k = 100, Q_x = 0.01, Q_v = 0.02, R = 2.0, P_0 = 1.0):
        # Continuou-time Matrix
        self.A_c = np.array([[   0.0,    1.0],
                             [-(k/m), -(c/m)]])
        
        self.B_c = np.array([[0.0],
                             [1/m]])
        
        # Discrete-time Matrix
        self.A = expm(self.A_c * dt)
        
        try:
            A_inv = np.linalg.inv(self.A_c)
        except:
            A_inv = np.linalg.pinv(self.A_c)

        self.B = A_inv @ (self.A - np.eye(2)) @ self.B_c

        self.C = np.array([[1.0, 0.0]])
        self.D = 0.0
        
        self.x_estimate = np.array( [[y_0], 
                                     [0.0]])
        
        self.Q = np.array([[Q_x, 0.0],
                           [0.0, Q_v]])
        self.R = np.array([[R]])
        self.P_estimate = np.eye(2) * P_0

    def estimate(self, y_measure, u):
        # Prediction
        self.x_predict = self.A @ self.x_estimate + self.B * u
        self.y_predict = self.C @ self.x_predict
        self.P_predict = self.A @ self.P_estimate @ self.A.T + self.Q
        
        # Update
        K = self.P_predict @ self.C.T @ np.linalg.inv(self.C @ self.P_predict @ self.C.T + self.R)
        self.x_estimate = self.x_predict + K @ (y_measure - self.y_predict)
        I = np.eye(self.A.shape[0])
        self.P_estimate = (I - K @ self.C) @ self.P_predict
        
        
if __name__ == "__main__":
    signal = pd.read_csv("01_Filter/Data/example08.csv")

    y_estimate = KalmanFilter(signal.y_measure[0],
                              Q_x = 0.001, Q_v = 0.01, R = 0.01, P_0 = 1.0)
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



