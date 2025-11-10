import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex06_TuningKalmanFilter import KalmanFilter

if __name__ == "__main__":
    
    signal = pd.read_csv("01_filter/Data/example06.csv")

    # Kalman Filter Parameter Set
    param_sets = [
        {"Q_x": 1e-4, "Q_v": 1e-3, "R": 0.5, "label": "Low Q"},
        {"Q_x": 1e-3, "Q_v": 1e-2, "R": 1.0, "label": "Baisc"},
        {"Q_x": 1e-1, "Q_v": 1e-1, "R": 5.0, "label": "High Q"},
        {"Q_x": 1e-4, "Q_v": 1e-1, "R": 0.1, "label": "Low R"}
    ]

    plt.figure(figsize=(10,6))
    plt.plot(signal.time, signal.y_measure,'k.', alpha=0.3, label="Measure")

    for p in param_sets:
        y_estimate = KalmanFilter(
            signal.y_measure[0],
            Q_x=p["Q_x"], Q_v=p["Q_v"], R=p["R"], P_0=1.0
        )
        signal["y_estimate_tmp"] = np.zeros_like(signal.y_measure)

        for i, row in signal.iterrows():
            y_estimate.estimate(signal.y_measure[i], signal.u[i])
            signal.loc[i, "y_estimate_tmp"] = y_estimate.x_estimate[1][0]

        plt.plot(signal.time, signal.y_estimate_tmp, label=p["label"])

    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.title('Kalman Filter Parameter Comparison')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
