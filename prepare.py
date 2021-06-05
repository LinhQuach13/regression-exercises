import pandas as pd
import numpy as np
import os
# acquire
from env import host, user, password
from pydataset import data

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

os.path.isfile('telco_df.csv')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
################################################## Telco Data ###########################################################################
# Use the above helper function and a sql query in a single function.
def new_telco_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    telco_sql = "SELECT * \
                 FROM customers \
                 JOIN contract_types USING(contract_type_id) \
                 JOIN internet_service_types USING(internet_service_type_id)\
                 JOIN payment_types USING(payment_type_id);"    
    return pd.read_sql(telco_sql, get_connection('telco_churn'))

def get_telco_data(cached=False):
    '''
    This function reads in telco churn data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('telco_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('telco_df.csv', index_col=0)
        
    return df




def telco_two_year():
    query= "Select customer_id, monthly_charges, tenure, total_charges FROM customers JOIN contract_types USING(contract_type_id) JOIN internet_service_types USING(internet_service_type_id)JOIN payment_types USING(payment_type_id) WHERE `contract_type_id` = '3';"
    df= pd.read_sql(query, get_connection('telco_churn'))
    return df




def clean_telco_data(df):
    '''
    This function take in the telco dataframe created and clean the total charges column
    by add a 0 to all empty columns and change the column to a float datatype.
    Returns new dataframe with total charges column cleaned.
    '''
    #Add zero to columns to convert to float
    df['total_charges'] = df['total_charges'] + '0'
    #make total charges into datatype float
    df['total_charges'] = df['total_charges'].astype('float')
    return df
    



def telco_split(df):
    '''
    This function take in the telco data acquired by get_telco_data,
    performs a split.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test



def wrangle_telco():
    df = clean_telco_data(get_telco_data())
    return telco_split (df)




def wrangle_telco_two_year():
    df = clean_telco_data(telco_two_year())
    return telco_split (df)



def scaler_telco():
    #Obtain train, validate, and test dataset using wrangle_telco function
    train,validate, test= wrangle_telco_two_year()
    #drop customer_id column
    train.drop(columns='customer_id', inplace= True) 
    validate.drop(columns='customer_id', inplace= True) 
    test.drop(columns='customer_id', inplace= True) 
    #Create the object
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(train[['monthly_charges']])
    x_train_scaled = scaler.transform(train[['monthly_charges']])
    x_validate_scaled = scaler.transform(validate)
    x_test_scaled = scaler.transform(test)
    return x_train_scaled, x_validate_scaled, x_test_scaled



################################# Zillow Data #########################################
def new_zillow_data():
    '''
    This function reads in Zillow data from CodeUp db and creates a dataframe
    '''
    zillow_query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
    FROM properties_2016 
    WHERE propertylandusetypeid = 261;'''
    return pd.read_sql(zillow_query, get_connection('zillow'))
     
    
    
def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df
        


def wrangle_zillow(df):
    '''
    Looks for existing zillow csv file and loads if present,
    otherwise runs new_zillow_data function to acquire data. Cleans nulls in zilllow dataframe
    '''
        
    # replace symbols, etc with NaN's
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # drop nulls
    df = df.dropna()
        
    return df
