"""
This class is intended to create a dataset from a time series [x1, x2, ..., xn]
for performing a regression task. The dataset follows this structure:

* Contains n - m + 1 rows, where n is the length of the series and m is the window size.
* Each instance consists of (Window_i, Target), where the number of predictors
  equals the window size.
* The Target is the element immediately after the window or at i + horizon steps ahead.
"""

import pandas as pd 
import numpy as np 

### Class to work just with sklearn 
class Window_Regressor_Sequence_to_scalar:
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
    
    
class Window_Regressor_Postional_Frequentist_Encoding:
    def __init__(self, time_series_data: pd.Series, window_size: int, horizon: int, freq_to_compute:int):
        self.time_series_data = time_series_data
        self.window_size = window_size
        self.horizon = horizon
        self.max_freq = freq_to_compute
        self.generated_data_set = self.generate_data_set()


    def compute_coefficients(self, zt, frequencies):
        T = len(zt)
        t = np.arange(T)
        
        omega_t = 2 * np.pi * frequencies.reshape(-1, 1) * t
        
        a_j = (2/T) * np.sum(zt.values * np.sin(omega_t), axis=1)
        b_j = (2/T) * np.sum(zt.values * np.cos(omega_t), axis=1)
        
        return a_j, b_j

    def periodogram(self):
        zt = self.time_series_data
        T = len(zt)
        num_freq = T//2 + 1
        frequencies = np.linspace(1/T, 0.1, num=num_freq, dtype=np.float32)
        
        a_j, b_j = self.compute_coefficients(zt, frequencies)
        periodogram = (T/2) * (np.square(a_j) + np.square(b_j))
        return dict(zip(frequencies, periodogram))

    def create_data_frame(self):
        # Firstly, we compute the top patient's frequencies
        frequencies_dict = self.periodogram()
        top_frequencies = sorted(frequencies_dict.items(), key=lambda x: x[1], reverse=True)[:self.max_freq]
        top_frequencies = {freq:(1/freq)/288 for freq, peridogram_value in top_frequencies} ## To get a dict freq :days
        self.top_frequencies = top_frequencies

        data = {}
        positions = self.time_series_data.index
        for freq, days in self.top_frequencies.items():
            c_i = 2 * np.pi * freq * positions
            sen_i = np.sin(c_i)
            cos_i = np.cos(c_i)
            
            data[f'f_{days}_sin'] = sen_i
            data[f'f_{days}_cos'] = cos_i
        data['glucose'] = self.time_series_data.values
        return pd.DataFrame(data)
    
    
    def generate_data_set(self):
        data_with_seasonal_columns = self.create_data_frame()
        n = self.time_series_data.shape[0]
        m = self.window_size
        data = []
        
        for i in range(n - (m+self.horizon) + 1):
            window_i = []
            for j in range(i, i + m):
                window_i.append(list(data_with_seasonal_columns.iloc[j].values))

            window_i = np.array(window_i)
            window_i = list(window_i.flatten())

            target_idx = j + self.horizon
            window_i.append(self.time_series_data.iloc[target_idx])
            data.append(window_i)
        
        num_predictors = m*data_with_seasonal_columns.shape[1]
        columns = [f'Predictor_{c}' for c in range(num_predictors)] + ['Target']
        return pd.DataFrame(data, columns=columns)
    


class Window_Regressor_Sequence_to_Sequence:
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
            for index_target_i in range(1, self.horizon+1):
                target_idx = j + index_target_i
                window_i.append(self.time_series_data.iloc[target_idx])
            data.append(window_i)
        
        columns = [f'Predictor_{c}' for c in range(m)] + [f'X_t+{h_i+1}' for h_i in range(self.horizon)]
        return pd.DataFrame(data, columns=columns)