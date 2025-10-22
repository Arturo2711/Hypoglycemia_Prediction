"""
This class is intended to create a dataset from a time series [x1, x2, ..., xn]
for performing a regression task. The dataset follows this structure:

* Contains n - m + 1 rows, where n is the length of the series and m is the window size.
* Each instance consists of (Window_i, Target), where the number of predictors
  equals the window size.
* The Target is the element immediately after the window or at i + horizon steps ahead.
"""

import pandas as pd 
from sklearn.model_selection import train_test_split

### Class to work just with sklearn 
class Window_Regressor:
    def __init__(self, time_series_data: pd.Series, window_size: int, horizon: int):
        self.time_series_data = time_series_data
        self.window_size = window_size
        self.horizon = horizon
        self.generated_data_set = self.generate_data_set()
    
    
    def generate_data_set(self):
        n = self.time_series_data.shape[0]
        m = self.window_size
        data = []
        
        for i in range(n - (m+self.horizon) + 1):
            window_i = []
            for j in range(i, i + m):
                window_i.append(self.time_series_data.iloc[j])
            target_idx = j + self.horizon
            window_i.append(self.time_series_data.iloc[target_idx])
            data.append(window_i)
        
        columns = [f'Predictor_{c}' for c in range(m)] + ['Target']
        return pd.DataFrame(data, columns=columns)