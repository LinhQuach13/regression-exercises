import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from env import host, user, password

# visualize
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(11, 9))
plt.rc('font', size=13)

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

import os
os.path.isfile('telco_df.csv')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import sklearn.preprocessing
from sklearn.preprocessing import QuantileTransformer



################################ Telco Data ###########################################################################################

def plot_variable_pairs(ds):
    '''
    This function takes in the telco train dataset and returns 2 lmplots (scatterplots with a regression line). 
    The first plot shows the relationship between tenure and total_charges. 
    The second plot shows the relationship between monthly_charges and total_charges.
    '''
    #lmplot of tenure with total_charges with tenure
    sns.lmplot(x="tenure", y="total_charges", data=ds, line_kws={'color': 'purple'})
    plt.show()
    #lmplot of tenure with total_charges with monthly_charges
    sns.lmplot(x="monthly_charges", y="total_charges", data=ds, line_kws={'color': 'purple'})
    plt.show();
    
    

    
def plot_variable_pairs2(ds):
    '''
    This function takes in the telco train dataset and returns 2 lmplots (scatterplots with a regression line). 
    The first plot shows the pairwise relationship between tenure, monthly_charges, and total_charges.
    - arguments:
    - ds: dataset or dataframe
    '''
    sns.pairplot(train[['tenure', 'monthly_charges', 'total_charges']], corner=True, kind= 'reg', plot_kws={'line_kws':{'color':'purple'}})
    plt.show();
    
    
    
def plot_quant(ds, cont_vars):
    '''
    This function takes in the train dataset, and continuous variable column list
    and ouputs the list as a plot.
    arguments: 
    - ds= dataset you want to input (typically the train dataset)
    - cont_vars= continuous variable list of columns
    '''
    #list of continuous variables
    cont_vars = ['monthly_charges', 'total_charges', 'tenure', 'tenure_years']
    for col in list(ds.columns):
            
            if col in cont_vars:
                sns.barplot(data = ds, y = col)
                plt.show()

def plot_cat(ds, cat_vars):
    '''
    This function takes in the train dataset, and categorical variable column list
    and ouputs the list as a plot.
    arguments: 
    - ds= dataset you want to input (typically the train dataset)
    - cat_vars= categorical variable list of columns
    '''
    #list of categorical variables
    cat_vars = ['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup','device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn', 'contract_type', 'internet_service_type', 'payment_type']
    for col in list(ds.columns):
        if col in cat_vars:
            sns.countplot(ds[col])
            plt.show()
            
            

def plot_categorical_and_continuous_vars(ds, cat_vars, cont_vars):
    
    '''
    This function takes in the train dataset, categorical variable column list, and continuous variable column list
    and ouputs the lists as plots.
    arguments: 
    - ds= dataset you want to input (typically the train dataset)
    - cat_vars= categorical variable list of columns
    - cont_vars= continuous variable list of columns
    '''
    
    plot_cat(ds, cat_vars)
    
    plot_quant(ds, cont_vars);
    
    
    
    
def months_to_years(ds):
    ds['tenure_years'] = ds.tenure / 12;
    