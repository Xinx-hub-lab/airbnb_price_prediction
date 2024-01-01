import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

import re



## functions for data cleaning
def encode_boolean(col):
    """
    col    : boolean series
    return : float series
    """
    return col.replace({'t': 1, 'f': 0}).astype(float)

def convert_to_category(col):
    """
    col    : object series
    return : category type series
    """
    return col.astype('category')

def one_hot_encoding(df, col):
    """
    df     : df 
    col    : col for one hot / category
    
    return : df concatenated with one hot 
    """
    df_encoded = pd.get_dummies(df[col], prefix='category').astype(float)
    df = pd.concat([df, df_encoded], axis=1)
    return df

def target_encoding(df, target, col):
    """
    df     : pd.df
    target : target col
    col    : categorical col
    return : median, imputed column
    """
    median_neigh = df.groupby(col)[target].median()
    return median_neigh, df[col].map(median_neigh)  

def target_encoding2(df, target, col, weight = 0.3):
    """
    df     : pd.df
    target : target col
    col    : categorical col
    return : global median, median, imputed column
    """
    median_global = df[target].median()
    median_neigh = df.groupby(col)[target].median()

    smooth = weight * median_neigh + (1-weight) * median_global
    return median_global, smooth, df[col].map(smooth).fillna(median_global)

def label_encoding(col):
    """
    col    : categorical col
    return : labeled col
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(col)

def get_days_since_date(col):
    """
    col    : object series
    return : float series
    """
    date_as_host = pd.to_datetime(col)
    return ((datetime.today() - date_as_host).dt.days).astype(float)





def outlier_to_na(df, col):
    """
    df     : df after first impute with no na
    col    : float series
    
    return : df after outlier kicked out
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR   ## 2.7σ (1.5) < 3σ
    upper_bound = Q3 + 1.5 * IQR
    if lower_bound < 0:
        lower_bound = 0
    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    return df, lower_bound, upper_bound

def mice_impute_numeric(df):
    """
    df     : whole df
    return : whole df
    """
    mice_imputer = IterativeImputer(imputation_order='ascending', random_state=42)
    df_numeric = df[[col for col in (df.columns).tolist() if df[col].dtype != 'object']]
    df_non_numeric = df[[col for col in (df.columns).tolist() if df[col].dtype == 'object']]
    imputed_values = mice_imputer.fit_transform(df_numeric)
    df_imputed = pd.concat([pd.DataFrame(imputed_values, columns=df_numeric.columns, index=df_numeric.index), 
                            df_non_numeric], 
                            axis=1)
    return df_imputed


def median_impute_numeric(col):
    """
    col    : numeric series
    return : median, imputed column
    """
    median = col.median()
    return median, col.fillna(median)

 
    
    

def count_luxury_amenities(amenities_list):
    luxury_amenities = {
    'Pool', 'Hot tub', 'Gym', 'Indoor fireplace', 'Private hot tub', 
    'Sauna', 'Piano', 'Beach essentials', 'Wine glasses', 'Private pool',
    'BBQ grill', 'Outdoor furniture', 'Fire pit', 'Private patio or balcony'}
    return sum(amenity in luxury_amenities for amenity in amenities_list)


def split_and_encode_verification(df):
    """
    replace verification col to 
    3 cols encoded
    """
    df['veri_email'] = ['email' in verification for verification in df['host_verifications']]
    df['veri_phone'] = ['phone' in verification for verification in df['host_verifications']]
    df['veri_work_email'] = ['work_email' in verification for verification in df['host_verifications']]
    df['veri_email'] = df.veri_email.replace({True: 1, False: 0})
    df['veri_phone'] = df.veri_phone.replace({True: 1, False: 0})
    df['veri_work_email'] = df.veri_work_email.replace({True: 1, False: 0})
    return df.drop(['host_verifications'], axis=1)



def parse_names(x):
    """
    x      : list of description info
    return : tuple of info
    """
    split_dat = x.split('·')
    rating, bedroom, bed, bath_num, bath_share = np.nan, np.nan, np.nan, np.nan, np.nan
    for dat in split_dat:
        dat = dat.strip()
        if '★' in dat:
            rating = dat.replace('★', '')
        elif 'bedroom' in dat:
            bedroom = (re.sub(r'[^\d.]+', '', dat)).strip()
        elif 'bed' in dat and 'bedroom' not in dat:
            bed = (re.sub(r'[^\d.]+', '', dat)).strip()
        elif 'bath' in dat:
            bath_num = (re.sub(r'[^\d.]+', '', dat)).strip()
            bath_share = 1 if 'share' in dat else 0
    return rating, bedroom, bed, bath_num, bath_share

def extract_name_column_info(df):
    """
    df: df with name col
    return: df of name info only, waiting for concat
    """
    processed_name_df = pd.DataFrame(df['name'].apply(lambda x: pd.Series(parse_names(x))))
    processed_name_df.columns = ['rating', 'bedrooms', 'beds', 'bath_num', 'bath_share']
    return processed_name_df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))




