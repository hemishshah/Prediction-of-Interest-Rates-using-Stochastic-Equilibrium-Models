import re
import numpy as np

class ratesData:

    def __init__(self,data,starting_column_name="",conversion=1,time_sort_columns:list=[]):

        self.starting_column_name = starting_column_name
        self.time_column_name = time_sort_columns
        self.conversion = conversion
        self.index_maturity_dict = self.extract_index_maturity(data,self.starting_column_name)
        self.maturity_columns = np.array(sorted(list(self.index_maturity_dict.values())))[:,1]
        self.maturity = np.array(sorted(list(self.index_maturity_dict.values())))[:,0]
        self.data = self.parse_data(data,self.maturity_columns,self.time_column_name)
        self.data = self.data.rename(columns=self.rename_columns_dict)


    def parse_data(self,df,rates_columns,time_columns):
        
        if time_columns:
            df.sort_values(by=time_columns,inplace=True)
        else:
            pass

        self.rates_data= df[rates_columns]*self.conversion

        return self.rates_data
        
    def extract_index_maturity(self,df,starting_column_name):

        # Extract columns that starts with the starting_column_name
        column_names = df.columns
        maturity_columns=[]
        rename_columns_dict ={}
        for column in column_names:
            if column.startswith(starting_column_name):
                maturity=float(re.findall("[-+]?(?:\d*\.*\d+)", column)[-1])
                maturity_columns.append([maturity,column])
                rename_columns_dict[column] = starting_column_name+str(maturity)
             
        self.rename_columns_dict = rename_columns_dict


        # Sort maturity_columns by maturity values
        maturity_columns = sorted(maturity_columns,key=lambda x: x[0])

        # Create Dict
        index_maturity_dict = {}
        for i in range(len(maturity_columns)):
            index_maturity_dict[i] = maturity_columns[i]

        return index_maturity_dict
