import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .utils import create_columns
from .ratesData import ratesData
from .InterestRateRegressor import InterestRateRegressor

class Vasiceck(InterestRateRegressor):

    def __init__(self,rates_data:ratesData,alpha=1,beta=1,sigma=0.5):
        super().__init__(rates_data)
        
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma


    def compute_A(self,maturity_array):
        """
        Coefficient A in Vasiceck Model
        """
        a = (self.sigma**2)/(2*(self.beta**2)) - (self.alpha/self.beta)
        b = (self.alpha/(self.beta**2)) - (self.sigma**2)/(self.beta**3)
        c = (self.sigma**2)/(4*(self.beta**3))
        A = np.exp(a*maturity_array + b*(1-np.exp(-self.beta*maturity_array) + c*(1-np.exp(-2*self.beta*maturity_array))))
        return A
        

    def compute_B(self,maturity_array):
        """
        Coefficient B in Vascicheck Model
        """
        return (1/self.beta)*(1-np.exp(-self.beta*maturity_array))


    def compute_short_rate(self,shortest_maturity,shortest_maturity_rates):

        """
        Compute short rate from the minimum maturirty data
        """
        shrt_mt_rt = np.array(shortest_maturity_rates)
        A_short_rate = self.compute_A(shortest_maturity)
        B_short_rate = self.compute_B(shortest_maturity)
        return (shrt_mt_rt*(shortest_maturity) + np.log(A_short_rate))/(B_short_rate)

    def compute_D_T(self,A,B,short_rate,t_array):
        """
        Function to calculate D(T) based on A and B
        """

        starting_column_name = "DT"

        D_T = []
        for i in short_rate:
            temp_d_t = np.multiply(A,np.exp(-B*i))
            D_T.append(temp_d_t)

        D_T_columns = create_columns(starting_column_name,t_array)
        D_T = pd.DataFrame(D_T,columns=D_T_columns)
        
        return ratesData(D_T,starting_column_name=starting_column_name) 

    def step(self):

        shortest_maturity = self.short_maturity
        shortest_maturity_column = self.rates_data.index_maturity_dict[0][1]
        shortest_maturity_rates = self.rates_data.data[shortest_maturity_column]
        short_rate = self.compute_short_rate(shortest_maturity,shortest_maturity_rates)

        A = self.compute_A(self.t_array)
        B = self.compute_B(self.t_array)
        D_T_data = self.compute_D_T(A,B,short_rate,self.t_array)
        par_model = self.compute_par_rates(D_T_data)
        return par_model


    def _fit(self,intial_values):

        self.alpha = intial_values[0]
        self.beta = intial_values[1]
        self.sigma = intial_values[2]
        return self.fit()
    
    def fit(self):
        par_model = self.step()
        diff = self.compute_diff(self.rates_data,par_model)
        return diff

    def optimize_func(self):
        result = least_squares(self._fit,[self.alpha,self.beta,self.sigma],loss='soft_l1')
        self.alpha,self.beta,self.sigma = result["x"][0],  result["x"][1], result["x"][2]
        return result
    

class Merton(InterestRateRegressor):

    def __init__(self,rates_data:ratesData,alpha=0.5,sigma=0.5):
        super().__init__(rates_data)
        
        self.alpha = alpha
        self.sigma = sigma


    def compute_short_rate(self,shortest_maturity,shortest_maturity_rates):

        """
        Compute short rate from the minimum maturirty data
        """
        shrt_mt_rt = np.array(shortest_maturity_rates)
        return shrt_mt_rt + ((self.alpha)/2)*shortest_maturity -  (self.sigma**2)*(shortest_maturity**2)/6

    def compute_D_T(self,short_rate,t_array):
        """
        Function to calculate D(T) based on A and B
        """

        starting_column_name = "DT"

        D_T = []
        for i in short_rate:
            temp_d_t = np.exp(-i*t_array - (self.alpha/2)*(np.power(t_array,2)) +  ((self.sigma)**2)*(np.power(t_array,3))/6)
            D_T.append(temp_d_t)

        D_T_columns = create_columns(starting_column_name,t_array)
        D_T = pd.DataFrame(D_T,columns=D_T_columns)
        
        return ratesData(D_T,starting_column_name=starting_column_name) 

    def step(self):

        shortest_maturity = self.short_maturity
        shortest_maturity_column = self.rates_data.index_maturity_dict[0][1]
        shortest_maturity_rates = self.rates_data.data[shortest_maturity_column]

        short_rate = self.compute_short_rate(shortest_maturity,shortest_maturity_rates)
        
        D_T_data = self.compute_D_T(short_rate,self.t_array)
        par_model = self.compute_par_rates(D_T_data)
        return par_model


    def _fit(self,values):
        self.alpha = values[0]
        self.sigma = values[1]
        return self.fit()
    
    def fit(self):
        par_model = self.step()
        diff = self.compute_diff(self.rates_data,par_model)
        return diff

    def optimize_func(self):
        result = least_squares(self._fit,[self.alpha,self.sigma],loss='soft_l1')
        self.alpha,self.sigma = result["x"][0], result["x"][1]
        return result
    

class Normal(InterestRateRegressor):

    def __init__(self,rates_data:ratesData,sigma=0.5):
        super().__init__(rates_data)

        self.sigma = sigma


    def compute_short_rate(self,shortest_maturity,shortest_maturity_rates):

        """
        Compute short rate from the minimum maturirty data
        """
        shrt_mt_rt = np.array(shortest_maturity_rates)
        return shrt_mt_rt -  (self.sigma**2)*(shortest_maturity**2)/6

    def compute_D_T(self,short_rate,t_array):
        """
        Function to calculate D(T) based on A and B
        """

        starting_column_name = "DT"

        D_T = []
        for i in short_rate:
            temp_d_t = np.exp(-i*t_array +  ((self.sigma)**2)*(np.power(t_array,3))/6)
            D_T.append(temp_d_t)

        D_T_columns = create_columns(starting_column_name,t_array)
        D_T = pd.DataFrame(D_T,columns=D_T_columns)
        
        return ratesData(D_T,starting_column_name=starting_column_name) 

    def step(self):

        shortest_maturity = self.short_maturity
        shortest_maturity_column = self.rates_data.index_maturity_dict[0][1]
        shortest_maturity_rates = self.rates_data.data[shortest_maturity_column]
        short_rate = self.compute_short_rate(shortest_maturity,shortest_maturity_rates)
        
        D_T_data = self.compute_D_T(short_rate,self.t_array)
        par_model = self.compute_par_rates(D_T_data)
        return par_model


    def _fit(self,intial_values):

        self.sigma = intial_values[0]
        return self.fit()
    
    def fit(self):
        par_model = self.step()
        diff = self.compute_diff(self.rates_data,par_model)
        return diff


    def optimize_func(self):
        result = least_squares(self._fit,[self.sigma],loss='soft_l1')
        return result