
import utils
import pandas as pd
import numpy as np
import ast



def train_clean_1_(df):
    
    median_values = {}
    bound_values = {}

    ## copy
    df_clean = df.copy()

    ## drop col and na row
    df_clean = df_clean.drop(['host_id', 'id', 'scrape_id', 'last_scraped', 'picture_url', 'calendar_last_scraped', 'host_name',
                              'description'], axis=1) ## 'neighbourhood_cleansed', 'property_type'
    df_clean = df_clean.dropna(subset = ['bathrooms_text'])

    ## encode 5 binary & impute 1var / host_is_superhost & save 
    bin_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable']
    for col in bin_cols:
        df_clean[col] = utils.encode_boolean(df_clean[col])
    super_median, imputed_col = utils.median_impute_numeric(df_clean['host_is_superhost'])
    df_clean['host_is_superhost'] = imputed_col
    median_values['host_is_superhost'] = super_median


    ## convert date
    df_clean['host_since'] = utils.get_days_since_date(df_clean['host_since'])



    ## impute*1: detect outliers
    outlier_cols = ['beds', 'accommodates', 
                    'minimum_nights', 'maximum_nights', 
                    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']
    for col in outlier_cols:
        df_clean, lower_bound, upper_bound = utils.outlier_to_na(df_clean, col)
        bound_values[col] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}

    ## impute*2: impute median
    for col in outlier_cols:
        median, df_clean[col] = utils.median_impute_numeric(df_clean[col])
        median_values[col] = median



    ## encode verification
    df_clean = utils.split_and_encode_verification(df_clean)

    ## count lux amenities
    df_clean['amenities'] = df_clean['amenities'].apply(ast.literal_eval)
    df_clean['lux_amen_count'] = df_clean['amenities'].apply(utils.count_luxury_amenities)
    df_clean = df_clean.drop(['amenities'], axis=1)


    ## name convert
    name_data = utils.extract_name_column_info(df_clean)
    df_clean.rename(columns={
        'beds': 'beds_provid'}, inplace=True)
    df_clean = pd.concat((df_clean, name_data), axis = 1)
    df_clean = df_clean.drop(['name', 'bathrooms_text', 'beds_provid'], axis=1)
    


    ## convert to category col
    cat_cols = ['neighbourhood_group_cleansed', 'room_type']
    for col in cat_cols:
        df_clean[col] = utils.label_encoding(df_clean[col])
        df_clean[col] = utils.convert_to_category(df_clean[col])

    # df_clean = df_clean.drop(cat_cols, axis=1)
    
    ## target encoding
    median_neigh, df_clean['neighbourhood_cleansed'] = utils.target_encoding(df_clean, 'price', 'neighbourhood_cleansed')
    median_prope, df_clean['property_type'] = utils.target_encoding(df_clean, 'price', 'property_type')
    
    ## impute 4 from name
    name_cols = ['rating', 'beds', 'bedrooms', 'bath_share','bath_num']
    for col in name_cols:
        median, df_clean[col] = utils.median_impute_numeric(df_clean[col])
        median_values[col] = median
    

    ## change order, price last
    feature_cols = [col for col in df_clean.columns if col != 'price']
    cols_order = feature_cols + ['price']

    # Reorder the DataFrame
    df_clean = df_clean[cols_order]

    return df_clean, median_values, bound_values, median_neigh, median_prope






## cleaning for test set
def test_clean_1_(df, median_dict, bound_dict, median_neigh, median_prope, test = True):

    ## copy
    df_clean = df.copy()
    ## drop col and na row
    df_clean = df_clean.drop(['host_id', 'id', 'scrape_id', 'last_scraped', 'picture_url', 'calendar_last_scraped', 'host_name',
                              'description'], axis=1)  ## 'neighbourhood_cleansed', 'property_type'

    ## encode 5 binary & no impute if train = True
    bin_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable']
    for col in bin_cols:
        df_clean[col] = utils.encode_boolean(df_clean[col])
    ## impute for host_is_superhost
    super_median = median_dict['host_is_superhost']
    df_clean['host_is_superhost'] = df_clean['host_is_superhost'].fillna(super_median)


    ## convert date
    df_clean['host_since'] = utils.get_days_since_date(df_clean['host_since'])


    ## impute*1: detect outliers
    outlier_cols = ['beds', 'accommodates', 
                    'minimum_nights', 'maximum_nights', 
                    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']
    for col in outlier_cols:
        lower_bound = bound_dict[col]['lower_bound']
        upper_bound = bound_dict[col]['upper_bound']
        df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
    ## impute*2: impute median
    for col in outlier_cols:
        median = median_dict[col]
        df_clean[col] = df_clean[col].fillna(median)
   

    ## encode verification
    df_clean = utils.split_and_encode_verification(df_clean)

    ## convert amenity
    df_clean['amenities'] = df_clean['amenities'].apply(ast.literal_eval)
    df_clean['lux_amen_count'] = df_clean['amenities'].apply(utils.count_luxury_amenities)
    df_clean = df_clean.drop(['amenities'], axis=1)
    

    ## name convert
    name_data = utils.extract_name_column_info(df_clean)
    df_clean.rename(columns={
        'beds': 'beds_provid'}, inplace=True)
    df_clean = pd.concat((df_clean, name_data), axis = 1)
    df_clean = df_clean.drop(['name', 'bathrooms_text', 'beds_provid'], axis=1)
    


    ## convert to category col
    cat_cols = ['neighbourhood_group_cleansed', 'room_type']
    for col in cat_cols:
        df_clean[col] = utils.label_encoding(df_clean[col])
        df_clean[col] = utils.convert_to_category(df_clean[col])
    df_clean['neighbourhood_cleansed'] = df_clean['neighbourhood_cleansed'].map(median_neigh)
    df_clean['property_type'] = df_clean['property_type'].map(median_prope)
    
    
    ## impute 4 from name
    name_cols = ['rating', 'beds', 'bedrooms', 'bath_share', 'bath_num']
    for col in name_cols:
        median = median_dict[col]
        df_clean[col] = df_clean[col].fillna(median)
    

    if test == False:
        ## change order, price last
        feature_cols = [col for col in df_clean.columns if col != 'price']
        cols_order = feature_cols + ['price']
        # Reorder the DataFrame
        df_clean = df_clean[cols_order]


    return df_clean







