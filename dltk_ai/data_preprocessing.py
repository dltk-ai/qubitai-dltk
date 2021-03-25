# Importing Libraries for performing Exploratory Data Analysis on dataset
import pandas as pd
from pandas_profiling import ProfileReport
#from autoviz.AutoViz_Class import AutoViz_Class
import dtale
import dtale.app as dtale_app


# Importing Sklearn modules for Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


# Importing Sklearn modules for Feature Transformation 
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer

# Importing Sklearn modules for Missing Value Imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import numpy as np


def read_data(file_name,header=0,sep=',',names=None,index_col=None,usecols=None,squeeze=False,mangle_dupe_cols=True,
              dtype=None,engine=None,converters=None,true_values=None,false_values=None,skiprows=None,
              skipfooter=0,nrows=None,na_values=None,keep_default_na=True,na_filter=True,verbose=False,
              parse_dates=False,date_parser=None,thousands=None,comment=None):
    """
    Parameters:
    file_name: Complete path of the data file
    sep: Delimiter to use for reading data
    header: Row number(s) to use as the column names. default 0 (first row)
    names: List of column names to use. array-like, optional.
    index_col: Column(s) to use as the row labels of the DataFrame. int, str, sequence of int / str, or False, default None.
    usecols: a list of subset of columns
    squeeze: If the parsed data only contains one column then return a Series. bool, default False
    mangle_dupe_cols: Duplicate columns will be specified with extension such as 1,2..etc. bool, default True
    dtype: specify datatype of particular column while reading eg: {‘a’: ‘Int64’}
    engine: Parser engine to use. {‘c’, ‘python’}, optional
    converters: Dict of functions for converting values in certain columns. optional.
    true_values: Values to consider as True. list, optional.
    false_values: Values to consider as False. list, optional.
    skiprows:Line numbers to skip. list-like, int or callable, optional.
    skipfooter: Number of lines at bottom of file to skip. int, default 0.
    nrows: Number of rows of file to read. int, default 0.
    na_values: Additional strings to recognize as NA/NaN. scalar, str, list-like, or dict, optional.
    keep_default_na: Whether or not to include the default NaN values when parsing the data. bool, default True.
    na_filter: Detect missing value markers. bool, default True.
    verbose: Indicate number of NA values placed in non-numeric columns. bool, default False.
    parse_dates: bool or list of int or names or list of lists or dict, default False
    date_parser: Function to use for converting a sequence of string columns to an array of datetime instances. function, optional.
    thousands: Thousands separator. str, optional.
    comment: Indicates remainder of line should not be parsed. str, optional.

    """
    file_type = 'csv' if file_name.endswith('.csv') else 'excel' if (file_name.endswith('.xlsx') or file_name.endswith('.xls')) else None
    if file_type == 'csv':
        return pd.read_csv(filepath_or_buffer=file_name,sep= sep,header=header,names=names,index_col=index_col,usecols=usecols, 
                           squeeze=squeeze,mangle_dupe_cols=mangle_dupe_cols,dtype=dtype,engine=engine,converters=converters,
                           true_values=true_values,false_values=false_values,skiprows=skiprows,skipfooter=skipfooter,nrows=nrows,
                           na_values=na_values,keep_default_na=keep_default_na,na_filter=na_filter,verbose=verbose,
                           parse_dates=parse_dates,date_parser=date_parser,thousands=thousands,comment=comment)
    elif file_type=='excel':
        if file_name.endswith('.xlsx'):
            engine = 'openpyxl'
        return pd.read_excel(io=file_name,header=header,names=names,index_col=index_col,usecols=usecols, squeeze=squeeze,
                             mangle_dupe_cols=mangle_dupe_cols,dtype=dtype,engine=engine,converters=converters,true_values=true_values,
                             false_values=false_values,skiprows=skiprows,skipfooter=skipfooter,nrows=nrows,na_values=na_values,
                             keep_default_na=keep_default_na,na_filter=na_filter,verbose=verbose,parse_dates=parse_dates,
                             date_parser=date_parser,thousands=thousands,comment=comment)


# Function for performing Exploratory Data Analysis on given dataset
def data_profile(dataframe,library="dtale",save_html=False,html_path=None):

    """
    This function is used for performing Exploratory Data Analysis for given dataset.

    Parameters:

    dataframe : dataset in the form of pandas Dataframe
    html_path: Path for saving html file created using Pandas-Profiling library
    save_html: Boolean parametr to create html report for EDA Profile using Pandas-Profiling library ; by default False
    Library: Takes the library for performing EDA out-of 3 libraries for given dataset;
             Valid parameter:[Pandas_profiling,Autoviz,DTale]
    
    Returns:
    Data Profiling report for given dataset
    """ 

   # To check if columns dataframe is not empty list 
    assert len(dataframe) > 0, "Please ensure that Dataframe is not empty"

    
    # To make sure user provide with Pandas Dataframe for EDA Libraries to execute
    assert (isinstance(dataframe, pd.DataFrame)),"Make sure Input is DataFrame"
    
    
    # Check supported libraries
    allowed_libraries = ["pandas-profiling", "autoviz", "dtale"]
    library = library.lower()
    assert library in allowed_libraries, f"Please select *Library* from {allowed_libraries}"

    if library =="pandas-profiling":
        if save_html==True:
            design_report= ProfileReport(dataframe)
            return design_report.to_file(output_file=html_path)
        elif save_html== False:
            design_report= ProfileReport(dataframe)
            return design_report.to_notebook_iframe()
        else:
            raise ValueError("Please provide required parameters")


    elif library == "autoviz":
        from autoviz.AutoViz_Class import AutoViz_Class
        AV = AutoViz_Class()
        return AV.AutoViz(filename="",dfte= dataframe)

    elif library == "dtale":
        return dtale.show(dataframe,ignore_duplicate=True)

    else:
        # If library is not chosen for performing EDA, function raise error
        raise ValueError("Please Select libraries from {allowed_libraries} for creating Data Profile of your Dataset")


# Function for Missing Value Imputation with given imputers: "Univariate_Imputation" and "Multivariate_Impuation" and in Multivariate Imputation two different method can be taken method=="Iterative Imputer" or method== KNN Imputer"

def impute_missing_value(dataframe,columns,imputer="univariate_imputation",method="iterative_imputer",replace=True,strategy='mean',
                        missing_values=np.nan,fill_value=None, verbose=0,copy=True, add_indicator=False,estimator=None,
                        sample_posterior=False,max_iter=10,tol=0.001,n_nearest_features=None,initial_strategy='mean',
                        imputation_order='ascending',skip_complete=False,min_value=None,max_value=None,random_state=None,
                        n_neighbors=5,weights='uniform',metric='nan_euclidean'):
  
  """
  This function is used for imputing Missing values of independent columns in the given dataset.

    Parameters:

    dataframe : dataset in the form of pandas Dataframe.
    columns : List of selected features from dataset for Imputing Missing Values.
    imputer : Imputers from sklearn-library for Missing Value Imputation;
              valid values: {"univariate_imputation","multivariate_Imputation"}.
    method :  Method for Multivariate Imputation; valid values: {"Iterative Imputer" and "KNN Imputer"}
    replace : Boolean parameter; if True replace original values with Dataframe and if False then transformed selected column
    missing_values: The placeholder for the missing values assigned to be np.nan as default for SimpleImputer function.
    strategy: The imputation strategy; it can be mean,median,most_frequent or constant value for SimpleImputer function.
    fill_value: When strategy == “constant”, fill_value is used to replace all occurrences of missing_values for SimpleImputer function.
    verbose: Controls the verbosity of the imputer.
    copy: boolean parameter, if True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
    add_indicator: If True, a MissingIndicator transform will stack onto output of the imputer’s transform. 
    estimator: The estimator to use at each step of the round-robin imputation where for IterativeImputer,by default None.
              valid values: {"BayesianRidge","DecisionTreeRegressor","ExtraTreesRegressor","KNeighborsRegressor","RandomForestRegressor"}
    sample_posterior: Boolean parameter to check whether to sample from the (Gaussian) predictive posterior of the fitted estimator for each imputation for IterativeImputer.
    max_iter: Maximum number of imputation rounds to perform before returning the imputations computed during the final round for IterativeImputer.
    tol: Tolerance of the stopping condition for IterativeImputer. 
    n_nearest_features: Number of other features to use to estimate the missing values of each feature column for IterativeImputer.
    initial_strategy: Which strategy to use to initialize the missing values for IterativeImputer;Valid values: {“mean”, “median”, “most_frequent”, or “constant”}
    imputation_order: The order in which the features will be imputed for IterativeImputer;Valid values: {“ascending”, “descending”, “roman”,"arabic" or “random”}
    skip_complete: Boolean parameter, if True then features with missing values during transform which did not have any missing values during fit will be imputed with the initial imputation method only for IterativeImputer.
    min_value: Minimum possible imputed value for IterativeImputer. 
    max_value: Maximum possible imputed value for IterativeImputer.
    random_state: The seed of the pseudo random number generator to use for IterativeImputer.
    n_neighbors: Number of neighboring samples to use for imputation for KNNImputer.
    weights: Weight function used in prediction for KNNImputer ;Valid values: {"uniform", "distance" or "callable"- a user-defined function which accepts an array of distances}
    metric: Distance metric for searching neighbors for KNNImputer; Possible values:{'nan_euclidean','callable'}


    Returns:
    Dataframe with Imputed Missing values.
  """

  assert (isinstance(dataframe, pd.DataFrame)),"Make sure Input is DataFrame"

  # To check if columns list is not empty list 
  assert len(columns) > 0, "Please ensure column names is not an empty list, select atleast 1 feature"
            
  # To check if columns list provided  by user exists in the dataframe
  for column in columns:
      assert column in dataframe.columns, "Please enter valid column name from the DataFrame"
            
  # Check supported imputers
  allowed_imputers = ["univariate_imputation","multivariate_imputation"]
  imputer  = imputer.lower()
  assert imputer  in allowed_imputers, f"Please select *Imputer* from {allowed_imputers}"


  if imputer == "univariate_imputation":
    imputer1 = SimpleImputer(strategy=strategy, missing_values= missing_values, fill_value=fill_value,
                            verbose=verbose, copy= copy, add_indicator=add_indicator) 
    imputer1.fit_transform(dataframe[columns])
    dataframe[columns] = imputer1.transform(dataframe[columns])
    if replace==True:
        final_df= dataframe
    else:
        final_df= dataframe[columns]
    return final_df

  elif imputer == "multivariate_imputation":

    # Check supported methods
    allowed_method = ["iterative_imputer","knn_imputer"]
    method  = method.lower()
    assert method in allowed_method, f"Please select *Method* from {allowed_method}"

    # Check supported estimators
    allowed_estimator = ["BayesianRidge","DecisionTreeRegressor","ExtraTreesRegressor","KNeighborsRegressor","RandomForestRegressor",None]
    assert estimator in allowed_estimator, f"Please select *Estimator* from {allowed_estimator}"

    if method=="iterative_imputer":
        if (estimator == "BayesianRidge" or estimator == None):
            imputer_estimator = BayesianRidge()
        elif estimator == "DecisionTreeRegressor":
            imputer_estimator = DecisionTreeRegressor()
        elif estimator == "ExtraTreesRegressor":
            imputer_estimator = ExtraTreesRegressor()
        elif estimator == "KNeighborsRegressor":
            imputer_estimator = KNeighborsRegressor()
        elif estimator == "RandomForestRegressor":
            imputer_estimator = RandomForestRegressor()
        else:
            raise ValueError("Select estimator for Multivariate Missing value Imputation to be applied on given Dataset")
            
        imputer2 = IterativeImputer(estimator=imputer_estimator,missing_values= missing_values,sample_posterior=sample_posterior,max_iter=max_iter,
        tol=tol,n_nearest_features=n_nearest_features,initial_strategy=initial_strategy,imputation_order=imputation_order,skip_complete=skip_complete,min_value=min_value,max_value=max_value,verbose=verbose,random_state=random_state,add_indicator=add_indicator)
        imputer2.fit_transform(dataframe[columns])
        dataframe[columns] = imputer2.transform(dataframe[columns])
        if replace==True:
            final_df= dataframe
        else:
            final_df= dataframe[columns]
        return final_df
      
    elif method=="knn_imputer":
      imputer3 = KNNImputer(missing_values=missing_values,n_neighbors=n_neighbors,weights=weights,metric=metric,copy=copy,add_indicator=add_indicator)
      imputer3.fit_transform(dataframe[columns])
      dataframe[columns] = imputer3.transform(dataframe[columns])
      if replace==True:
        final_df= dataframe
      else:
          final_df= dataframe[columns]
      return final_df
        
    else:
        raise ValueError("Select method for Multivariate Missing value Imputation to be applied on given Dataset")

  else:
    raise ValueError("Select method for Missing-value to be applied on given Dataset")



def treat_outliers(dataframe, column, **kwargs):
    """
    Parameters:

    dataframe: Pandas DataFrame
    column: list of columns/column to treat outliers
    **kwrags - Default - Statistic - mean
        select any one of the below parameters - statistic/value/remove
        remove : True/False - flag to delete the rows with outliers
        statistic - min/max/mean/median/quantilevalue
        value - replaces outliers with user specified value
    """
    dataframe_duplicate = dataframe.copy()
    # params in kwargs
    function_args_list = ['remove', 'value', 'statistic', 'higher', 'lower']
    user_input_args_list = list(kwargs.keys())
    args_check_flag = all(elem in function_args_list for elem in user_input_args_list)

    if args_check_flag == False:
        raise ValueError('please choose a valid argument')
    datatype_check_flag = all(
        [True if (dataframe_duplicate[i].dtypes == 'int64' or dataframe_duplicate[i].dtypes == 'float64') else False for
         i in column])
    if not datatype_check_flag:
        raise Exception('function not applicable for columns of type category')

    missing_value_check_flag = all([True if dataframe_duplicate[i].isnull().sum() > 1 else False for i in column])
    if missing_value_check_flag:
        raise ValueError('Contains missing values - please use missing_data function to fill missing values')

    remove = kwargs.get('remove', False)  # takes the given value of remove flag

    upper = kwargs.get('upper', True)

    lower = kwargs.get('lower', True)

    new_column = []
    for i in column:

        higher_outlier = dataframe_duplicate[i].quantile(0.75) + 1.5 * (
                dataframe_duplicate[i].quantile(0.75) - dataframe_duplicate[i].quantile(0.25))

        lower_outlier = dataframe_duplicate[i].quantile(0.25) - 1.5 * (
                dataframe_duplicate[i].quantile(0.75) - dataframe_duplicate[i].quantile(0.25))

        if remove:

            if kwargs['remove'] not in [True, False]: raise ValueError('remove should be either True/False')

            if upper:
                dataframe_duplicate = dataframe_duplicate[dataframe_duplicate[i] <= higher_outlier]
            if lower:
                dataframe_duplicate = dataframe_duplicate[dataframe_duplicate[i] >= lower_outlier]
        else:
            if 'value' in kwargs:
        
                if type(kwargs['value']) == str: 
                    raise ValueError('value should be a integer or float')
                value = kwargs['value']
            elif 'statistic' in kwargs:
                if kwargs['statistic'] in ['min', 'max', 'mean', 'median'] or type(kwargs['statistic']) == int or type(kwargs['statistic']) == float:
                    pass
                else:
                    raise ValueError('statistic should be min/max/mean/median or a quantile value between 0 and 1')
                minimum = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].min()
                maximum = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].max()
                mean = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].mean()
                median = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].median()
                quantile_value = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].quantile(
                    kwargs['statistic']) if (
                        type(kwargs['statistic']) == int or type(kwargs['statistic']) == float) else 0

                value = minimum if kwargs['statistic'] == 'min' else maximum if kwargs[
                                                                                    'statistic'] == 'maximum' else mean if \
                    kwargs['statistic'] == 'mean' else median if kwargs[
                                                                     'statistic'] == 'median' else quantile_value if type(
                    kwargs['statistic']) == int or type(kwargs['statistic']) == float else 0
            else:
                value = dataframe_duplicate[i][
                    (dataframe_duplicate[i] > lower_outlier) & (dataframe_duplicate[i] < higher_outlier)].mean()

            if upper:
                dataframe_duplicate[i] = np.where(dataframe_duplicate[i] >= higher_outlier, value,
                                                  dataframe_duplicate[i])
            if lower:
                dataframe_duplicate[i] = np.where(dataframe_duplicate[i] <= lower_outlier, value,
                                                  dataframe_duplicate[i])

    return dataframe_duplicate



# Function for Converting datatype of the dataset
def convert_dtypes(dataframe,column_datatypes):

    """
    This function is used for converting Datatype as per user requirement for given dataset.

    Parameters:

    dataframe : dataset in the form of pandas Dataframe
    column_datatypes: Takes dictionary as input where keys are column_names and values are required datatypes for conversion.
                      Example: dltk_ai.convert_dtypes(df,{"GarageYrBlt":"int64"})

    Returns:
    Changed Datatype of Dataframe columns
    """ 

    # To make sure user provide with Pandas Dataframe 
    assert (isinstance(dataframe, pd.DataFrame)),"Make sure Input is DataFrame"

    # To check if columns dctionary is not empty list 
    assert len(column_datatypes) > 0, "Please ensure that key-value pair is not empty dictionary, select atleast 1 key-value pair"


    column_names = list(column_datatypes.keys())
    for column in column_names:
        dataframe[column] = dataframe[column].astype(column_datatypes[column])
    print("Changed datatypes of Dataframe columns \n",dataframe.dtypes)
    #return dataframe




# Function for Feature Scaling with given methods:"MinMaxScaler","StandardScaler","MaxAbsScaler","RobustScaler",and "Normalizer"

def feature_scaling(dataframe,column_names,method="minmaxscaler",replace=True,feature_range=(0,1),copy=True,clip=False,use_mean=True,
                    use_std=True,use_centering=True, use_scaling=True, quantile_range=(25.0, 75.0),unit_variance=False,norm='l2'):
    
    """
    This function is used to normalize the range of independent features in dataset.

    Parameters:

    dataframe : dataset in the form of pandas Dataframe for all methods
    column_names : User selected features from dataset for Scaling
    method : Methods from sklearn-library for Feature Scaling; valid values:{"MinMaxScaler","StandardScaler","MaxAbsScaler","RobustScaler",and "Normalizer"}.
    replace : Boolean parameter; if True replace original values with Dataframe,otherwise transformed selected column
    feature_range: user-defined range for scaling features given in form of tuple for MinMaxScaler method; by default feature scaled in range of (0,1)
    copy: boolean parameter, if true creates copy and then scale the variables otherwise inplaced in same dataframe.
    clip: Set to True to clip transformed values of held-out data to provided feature range.
    use_mean: Boolean parameter,if True then center the data before scaling for StandardScaler method.
    use_std: Boolean parameter,If True then scale the data to unit variance for StandardScaler method.
    use_centering: parameter for RobustScaler method
    use_scaling: Boolean parameter ,If True then scale the data to interquartile range for RobustScaler method.
    quantile_range: Quantile range used for scaling the selected set of features in RobustScaler method.
    unit_variance: Boolean parameter,If True then scale data so that normally distributed features have a variance of 1 used in RobustScaler method.
    norm: Used to normalize selected columns that are non zero sample for Normalizer method.

    Returns:
    Dataframe with scaled features
    """

    # To make sure user provide with Pandas Dataframe 
    assert (isinstance(dataframe, pd.DataFrame)),"Make sure Input is DataFrame"

    # To check if columns list is not empty list 
    assert len(column_names) > 0, "Please ensure column names is not an empty list, select atleast 1 feature"
    
    # To check if columns list provided  by user exists in the dataframe
    for column in column_names:
        assert column in dataframe.columns, "Please enter valid column name from the DataFrame"
      
    # Check supported methods
    allowed_methods = ["minmaxscaler","standardscaler","maxabsscaler","robustscaler","normalizer"]
    method = method.lower()
    assert method in allowed_methods, f"Please select *Scaling_Method* from {allowed_methods}"

    # Assertion for normalizer
    allowed_norm = ["l1","l2"]
    norm = norm.lower()
    assert norm in allowed_norm, f"Please select *norm* from {allowed_norm}"

    df_scaled = dataframe.copy()
    features = df_scaled[column_names]
    
    if method=="minmaxscaler":
        scaler1 = MinMaxScaler(feature_range,copy=copy)
        df_scaled[column_names] = scaler1.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    elif method=="standardscaler": 
        scaler2 = StandardScaler(copy=copy, with_mean= use_mean, with_std= use_std)
        df_scaled[column_names] = scaler2.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    elif method=="maxabsscaler": 
        scaler3 = MaxAbsScaler(copy=copy)
        df_scaled[column_names] = scaler3.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    elif method=="robustscaler": 
        scaler4 = RobustScaler(with_centering= use_centering, with_scaling= use_scaling, quantile_range=quantile_range, copy=copy)
        df_scaled[column_names] = scaler4.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    elif method=="normalizer":
        scaler5 = Normalizer(copy=copy,norm=norm)
        df_scaled[column_names] = scaler5.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    else:
        raise ValueError("Please select method from {allowed_methods} for Feature scaling")


# Function for Feature Transformation with given methods:"Quantile Transformer","Power Transformation" and "Custom Transformation"

def feature_transformation(dataframe,column_names,transformer="quantile_transformer",copy=True,output_distribution='uniform',n_quantiles=1000,
                            ignore_implicit_zeros=False,subsample=1e5,random_state=None,method='yeo-johnson',standardize=True,func=None,
                            inverse_func=None, validate=False, accept_sparse=False,check_inverse=True, kw_args=None, inv_kw_args=None,
                            replace=True):
      
    """
    This function is used for transforming set of independent feature in given dataset.

    Parameters:

    dataframe : dataset in the form of pandas Dataframe.
    column_names : List of features from dataset for Transformation.
    transformer : Methods from sklearn-library for Feature Transformation; valid values:{"Quantile Transformer","Power Transformation","Custom Transformation"}
    replace : Boolean parameter; if True replace original values with Dataframe,otherwise transformed selected column
    copy: boolean parameter, if true creates copy and then scale the variables otherwise inplaced in same dataframe.
    output_distribution: For marginal distribution for the transformed data used in Quantile Transformer; {'uniform', 'normal'}
    n_quantiles: Number of quantiles to be computed used in Quantile Transformer where default values is 1000
    ignore_implicit_zeros: Boolean parameter used for sparse matrices in Quantile Transformer 
    subsample: Maximum number of samples used to estimate the quantiles for computational efficiency in Quantile Transformer; default is 1e5
    random_state: Determines random number generation for subsampling and smoothing noise in Quantile Transformer.
    method: Used in power transformers taking ‘yeo-johnson’ or ‘box-cox’ as parameter for Power Transformer.
    standardize:Boolean parameter when set to True to apply zero-mean, unit-variance normalization to the transformed output for Power Transformer.
    func: The callable to use for the transformation in FunctionTransformer. By default its None then func will be the identity function.
    inverse_func: The callable to use for the inverse transformation in Custom Transformation.By default its None then func will be the identity function for FunctionTransformer
    validate: Boolean parameter, to indicate that the input X array should be checked before calling func for FunctionTransformer.
    accept_sparse: Boolean parameter, to indicate that func accepts a sparse matrix as input for FunctionTransformer.
    check_inverse: Boolean parameter, to check whether func followed by inverse_func leads to the original inputs for FunctionTransformer.
    kw_args: Dictionary of additional keyword arguments to pass to func for FunctionTransformer.
    inv_kw_args: Dictionary of additional keyword arguments to pass to inverse_func for FunctionTransformer.

    Returns:
    DataFrame with Transformed variables
    
    """

    # To make sure user provide with Pandas Dataframe
    assert (isinstance(dataframe, pd.DataFrame)),"Make sure Input is DataFrame"

    # To check if columns list is not empty list 
    assert len(column_names) > 0, "Please ensure column names is not an empty list, select atleast 1 feature"
    
    # To check if columns list provided  by user exists in the dataframe
    for column in column_names:
        assert column in dataframe.columns, "Please enter valid column name from the DataFrame"

    # Check supported transformers
    allowed_transformer = ["quantile_transformer","power_transformer","custom_transformer"]
    transformer  = transformer.lower()
    assert transformer in allowed_transformer, f"Please select *Transformation_Method* from {allowed_transformer}"

    df_scaled = dataframe.copy()
    features = df_scaled[column_names]

    if transformer=="quantile_transformer":
        transformer1 = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, 
                                           ignore_implicit_zeros=ignore_implicit_zeros,
                                           subsample=subsample, random_state=random_state, copy= copy)
        df_scaled[column_names] = transformer1.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    if transformer=="power_transformer": 
        transformer2 = PowerTransformer(method=method,standardize=standardize)
        df_scaled[column_names] = transformer2.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    elif transformer=="custom_transformer":
        transformer3 = FunctionTransformer(func, inverse_func=inverse_func, validate=validate, accept_sparse=accept_sparse, check_inverse=check_inverse, kw_args=kw_args, inv_kw_args=inv_kw_args)
        df_scaled[column_names] = transformer3.fit_transform(features.values)
        if replace==True:
            final_df= df_scaled
        else:
            final_df= df_scaled[column_names]
        return final_df

    else:
        raise ValueError("Please select Transformer from {allowed_transformer} for Feature Transformation")

def feature_creation(dataframe, feature_method, binning_column=None, bins=10, binning_right=True, binning_labels=None,
                     binning_retbins=False, binning_precision=3, binning_include_lowest=False,
                     binning_duplicates='raise', binning_ordered=True, dummies_prefix=None, dummies_prefix_sep='_',
                     dummies_dummy_na=False, dummies_column=None, dummies_sparse=False, dummies_drop_first=False,
                     groupby_column=None, groupby_transform_column=None, groupby_transform_metric='mean'):
    """
    Parameters:
    dataframe
    feature_method: binning/one-hot-encoding/groupby
    params if feature_method is binning: The criteria to bin by.
        bins - default int 10 - 
            int : Defines the number of equal-width bins in the range of x. The range of x is extended by .1% on each side to include the minimum and maximum values of x
            sequence of scalars : Defines the bin edges allowing for non-uniform width. No extension of the range of x is done.
            IntervalIndex : Defines the exact bins to be used.
        binning_right: bool, default True
            Indicates whether bins includes the rightmost edge or not. If right == True (the default), then the bins [1, 2, 3, 4] indicate (1,2], (2,3], (3,4]. This argument is ignored when bins is an IntervalIndex.
        binning_labels: array or bool, optional
            Specifies the labels for the returned bins. Must be the same length as the resulting bins. If False, returns only integer indicators of the bins. This affects the type of the output container (see below). This argument is ignored when bins is an IntervalIndex.
        binning_retbins: bool, default False
            Whether to return the bins or not. Useful when bins is provided as a scalar.
        binning_precision:  int, default 3
            The precision at which to store and display the bins labels.
        binning_include_lowest: bool, default False
            Whether the first interval should be left-inclusive or not.
        binning_duplicates: {default ‘raise’, ‘drop’}, optional
            If bin edges are not unique, raise ValueError or drop non-uniques.
    params if feature_method is one-hot-encoding: Convert categorical variable into dummy/indicator variables.
        dummies_prefix - str, list of str, or dict of str, default None
            String to append DataFrame column names. Pass a list with length equal to the number of columns when calling get_dummies on a DataFrame. Alternatively, prefix can be a dictionary mapping column names to prefixes.
        dummies_prefix_sep: str, default ‘_’
            If appending prefix, separator/delimiter to use. Or pass a list or dictionary as with prefix.
        dummies_dummy_na: bool, default False
            Add a column to indicate NaNs, if False NaNs are ignored.
        dummies_column: list-like, default None
            Column names in the DataFrame to be encoded. If columns is None then all the columns with object or category dtype will be converted.
        dummies_sparse: bool, default False
            Whether the dummy-encoded columns should be backed by a SparseArray (True) or a regular NumPy array (False).
        dummies_drop_first: bool, default False
            Whether to get k-1 dummies out of k categorical levels by removing the first level.
    params if feature_method is groupby: groups by the specified columns, gets the metrics of the transform column in each group. Returns a new column
        groupby_column - list of column/columns
        groupby_transform_column - column to apply the metric on 
        groupby_transform_metric - min/max/mean/median
    """
    if feature_method == 'binning':
        # if binning is selected and binning column is not specified - raise error
        if binning_column is None:
            raise ValueError('please specify a column for binning')
        else:
            # call pandas pd.cut function with user specified params
            return pd.cut(dataframe[binning_column], bins=bins, right=binning_right, labels=binning_labels,
                          retbins=binning_retbins, precision=binning_precision, include_lowest=binning_include_lowest,
                          duplicates=binning_duplicates)
    elif feature_method == 'one-hot-encoding':
        # call get_dummies
        return pd.get_dummies(dataframe, prefix=dummies_prefix, prefix_sep=dummies_prefix_sep,
                              dummy_na=dummies_dummy_na,
                              columns=dummies_column, sparse=dummies_sparse, drop_first=dummies_drop_first)
    elif feature_method == 'groupby':
        if groupby_column is None or groupby_transform_column is None:
            raise ValueError('please specify a valid groupby_column/groupby_transform_column')
        group_column = groupby_column if type(groupby_column) == list else [groupby_column]
        return dataframe.groupby(group_column)[groupby_transform_column].transform(groupby_transform_metric)
    else:
        pass


def data_transformation(dataframe, transform_method, pivot_index=None, pivot_columns=None, pivot_values=None,
                        melt_id_vars=None, melt_value_vars=None, crosstab_columns=None, crosstab_rows=None):
    """
    Parameters
    dataframe
    transform_method: pivot/melt/crosstab
    params if transform_method is pivot
        Return reshaped DataFrame organized by given index / column values.

        Reshape data (produce a “pivot” table) based on column values. Uses unique values from specified index / columns to form axes of the resulting DataFrame. This function does not support data aggregation, multiple values will result in a MultiIndex in the columns. See the User Guide for more on reshaping.

        pivot_index: str or object or a list of str, optional
            Column to use to make new frame’s index. If None, uses existing index.

        pivot_columns: str or object or a list of str
            Column to use to make new frame’s columns.

        pivot_values: str, object or a list of the previous, optional
            Column(s) to use for populating new frame’s values. If not specified, all remaining columns will be used and the result will have hierarchically indexed columns.
    params if transform_method is melt:
        Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

        This function is useful to massage a DataFrame into a format where one or more columns are identifier variables (id_vars), while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.

        melt_id_vars: tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        melt_value_vars: tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.

    params if transform_method is crosstab
        crosstab_rows: array-like, Series, or list of arrays/Series
            Values to group by in the rows.
        crosstab_columns: array-like, Series, or list of arrays/Series.
            Values to group by in the columns.
        """
    if transform_method not in ['pivot', 'melt', 'crosstab']:
        raise ValueError('transform_method should be either pivot/melt/crosstab')
    if transform_method == 'pivot':
        if pivot_index is None or pivot_columns is None:
            raise ValueError('pivot_index and pivot_columns are needed for transform_method pivot')
        else:
            return dataframe.pivot(index=pivot_index, columns=pivot_columns, values=pivot_values)
    elif transform_method == 'melt':
        if melt_id_vars is None or melt_value_vars is None:
            raise ValueError('melt_id_vars and melt_value_cars are needed for transform_method melt')
        else:
            return dataframe.melt(id_vars=melt_id_vars, value_vars=melt_value_vars)
    elif transform_method == 'crosstab':
        if crosstab_columns is None or crosstab_rows is None:
            raise ValueError('crosstab_columns and crosstab_rows are needed for transform_method crosstab')
        else:
            crosstab_rows = crosstab_rows if type(crosstab_rows) != list else crosstab_rows[0]
            crosstab_columns = crosstab_columns if type(crosstab_columns) != list else crosstab_columns[0]
            return pd.crosstab(dataframe[crosstab_rows], dataframe[crosstab_columns])
    else:
        pass



