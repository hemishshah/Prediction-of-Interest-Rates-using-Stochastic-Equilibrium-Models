from abc import abstractmethod
import numpy as np
import pandas as pd
from .ratesData import ratesData



class InterestRateRegressor:

    def __init__(self,rates_data:ratesData,coupon_frequency="mid"):

        self.rates_data = rates_data

        if coupon_frequency=="mid":
            self.coupon_frequency = 0.5
        elif coupon_frequency=="quaterly":
            self.coupon_frequency = 0.25
        elif coupon_frequency=="yearly":
            self.coupon_frequency = 1

        self.short_maturity = float(self.rates_data.maturity[0])
        self.maturities = np.asarray(self.rates_data.maturity[1:], dtype=float)

        self.t_start = self.coupon_frequency
        self.t_end = np.max(self.maturities) + self.coupon_frequency
        self.t_array = np.arange(self.t_start,self.t_end,self.coupon_frequency)


    @abstractmethod
    def compute_DT(self,*args,**kwrgs)-> ratesData:
        """Compute Discount Factor Matrix"""
        ...

    @abstractmethod
    def optimize_func(self,**kwargs):
        """
        Run objective function on scipy's optimize function
        """
        ...

    @abstractmethod
    def step(self,**kwargs):
        """
        Function to compute one training step
        """
        ...

    @abstractmethod
    def predict(self,short_rate):
        """
        Function to predict in maturity rates according to short rates
        """
        ...

    
    def compute_par_rates(self,DT_data:ratesData)-> ratesData:
        """
        Compute Par Rates according to frequency
        """
        df_par_model = pd.DataFrame()
        for t in self.maturities:

            column_names = [DT_data.starting_column_name + str(i) for i in np.arange(self.t_start,t+self.coupon_frequency,self.coupon_frequency)]
            df_i = DT_data.data[column_names]

            temp_par = []
            for _,row in df_i.iterrows():
                par_rate = 2*(1-1*row[-1])/(np.sum(np.array(row)))
                temp_par.append(par_rate)

            df_par_model['par_' + str(t)] = pd.Series(temp_par)

        return ratesData(df_par_model,starting_column_name="par_",conversion=1)
    
    def compute_diff(self,par_actual:ratesData,par_model:ratesData):
        """
        Difference between the actual par values and compted par values
        """
        
        # Extract maturities
        act_maturty= par_actual.maturity
        modl_maturty = par_model.maturity

        # Create columns based on common maturity
        common_maturty = set(act_maturty).intersection(modl_maturty)
        par_act_cols = [par_actual.starting_column_name + str(i) for i in common_maturty]
        par_modl_cols = [par_model.starting_column_name + str(i) for i in common_maturty]
        return par_model.data[par_modl_cols].to_numpy().flatten()-par_actual.data[par_act_cols].to_numpy().flatten()
   