*******************
Data Preprocessing
*******************

DLTK's Data Preprocessing consists of basic pre-processing techniques used before model building.

Read Data
---------

Function for reading data from csv or excel to dataframe.

.. function:: preprocessor.read_data(file_name,header=0,names=None,index_col=None,usecols=None,
                                    squeeze=False,mangle_dupe_cols=True,dtype=None,engine=None,
                                    converters=None,true_values=None,false_values=None,skiprows=None,
                                    skipfooter=0,nrows=None,na_values=None,keep_default_na=True,
                                    na_filter=True,verbose=False,parse_dates=False,date_parser=None,
                                    thousands=None,comment=None)


    :param data_path: Path for data file - csv or excel.
    :param file_name: Complete path of the data file.
    :param header: Row number(s) to use as the column names. default 0 (first row).
    :param names: List of column names to use. array-like, optional.
    :param index_col: Column(s) to use as the row labels of the DataFrame. int, str, sequence of int / str, or False, default None.
    :param usecols: A list of subset of columns.
    :param squeeze: If the parsed data only contains one column then return a Series. bool, default False.
    :param mangle_dupe_cols: Duplicate columns will be specified with extension such as 1,2..etc. bool, default True.
    :param dtype: specify datatype of particular column while reading eg: {‘a’: ‘Int64’}.
    :param engine: Parser engine to use. {'c', 'python'}, optional.
    :param converters: Dict of functions for converting values in certain columns. optional.
    :param true_values: Values to consider as True. list, optional.
    :param false_values: Values to consider as False. list, optional.
    :param skiprows: Line numbers to skip. list-like, int or callable, optional.
    :param skipfooter: Number of lines at bottom of file to skip. int, default 0.
    :param nrows: Number of rows of file to read. int, default 0.
    :param na_values: Additional strings to recognize as NA/NaN. scalar, str, list-like, or dict, optional.
    :param keep_default_na: Whether or not to include the default NaN values when parsing the data. bool, default True.
    :param na_filter: Detect missing value markers. bool, default True.
    :param verbose: Indicate number of NA values placed in non-numeric columns. bool, default False.
    :param parse_dates: Bool or list of int or names or list of lists or dict, default False.
    :param date_parser: Function to use for converting a sequence of string columns to an array of datetime instances. function, optional.
    :param thousands: Thousands separator. str, optional.
    :param comment: Indicates remainder of line should not be parsed. str, optional.

.. note:: Supports csv, xls & xlsx formats only.

**Example**::

    import dltk_ai

    df = dltk_ai.read_data("diabetes_train.csv")


Data Profile
------------
Function for performing Exploratory Data Analysis on given dataset.

*Supported Libraries*

    1.Pandas-Profiling
    2.AutoViz
    3.DTale



.. function:: data_profile(dataframe,library='dtale',save_html,html_path)

    :param dataframe: Pandas Dataframe.
    :param Library: Library for performing EDA out-of 3 libraries for given dataset.
    :param save_html: Boolean parametr to create html report for EDA Profile using Pandas-Profiling library ; by default False
    :param html_path: Path for saving html file created using Pandas-Profiling library

    :rtype: Data Profiling report for given dataset.

.. note:: Consider dataset in Dataframe only.

**Example**::

    import dltk_ai

    df = dltk_ai.read_data("diabetes_train.csv")

    dltk_ai.data_profile(df,"pandas-profiling")


Missing Value Imputation
------------------------
Function is used for imputing Missing values for selected set of independent columns in the given dataset.

.. function:: impute_missing_value(dataframe,columns,imputer,method="iterative_imputer",replace=True,
                                   strategy='mean',missing_values=np.nan,fill_value=None,verbose=0,
                                   copy=True, add_indicator=False,estimator=None,sample_posterior=False,
                                   max_iter=10,tol=0.001,n_nearest_features=None,initial_strategy='mean',
                                   imputation_order='ascending',skip_complete=False,min_value=None,
                                   max_value=None,random_state=None,n_neighbors=5,weights='uniform',
                                   metric='nan_euclidean'):

    :param dataframe: Pandas Dataframe.
    :param columns: List of selected features from dataset for Imputing Missing Values.
    :param imputer: Imputers from sklearn-library for Missing Value Imputation;
                    valid values: {"Univariate_imputation","Multivariate_Imputation"}.
    :param method:  Method for Multivariate Imputation; valid values: {"Iterative_Imputer" and "KNN_Imputer"}.
    :param replace: Boolean parameter,if True replace original values with Dataframe and if False then transformed selected column.
    :param missing_values: The placeholder for the missing values assigned to be np.nan as default for SimpleImputer function.
    :param strategy: The imputation strategy; it can be mean,median,most_frequent or constant value for SimpleImputer function.
    :param fill_value: When strategy == “constant”, fill_value is used to replace all occurrences of missing_values for SimpleImputer function.
    :param verbose: Controls the verbosity of the imputer.
    :param copy: Boolean parameter, if True, a copy of X will be created. If False, imputation will be done in-place whenever possible.
    :param add_indicator: If True, a MissingIndicator transform will stack onto output of the imputer’s transform. 
    :param estimator: The estimator to use at each step of the round-robin imputation for IterativeImputer,by default None.
                   valid values: {"BayesianRidge","DecisionTreeRegressor","ExtraTreesRegressor","KNeighborsRegressor","RandomForestRegressor"}
    :param sample_posterior: Boolean parameter to check whether to sample from the (Gaussian) predictive posterior of the fitted estimator 
                             for each imputation for IterativeImputer.
    :param max_iter: Maximum number of imputation rounds to perform before returning the imputations computed during the final round 
                     for IterativeImputer.
    :param tol: Tolerance of the stopping condition for IterativeImputer. 
    :param n_nearest_features: Number of other features to use to estimate the missing values of each feature column for IterativeImputer.
    :param initial_strategy: Strategy to use to initialize the missing values for IterativeImputer
                             Valid values: {“mean”, “median”, “most_frequent”, or “constant”}.
    :param imputation_order: Order in which the features will be imputed for IterativeImputer
                             Valid values: {“ascending”, “descending”, “roman”,"arabic" or “random”}
    :param skip_complete: Boolean parameter, if True then features with missing values during transform which did not have any 
                          missing values during fit will be imputed with the initial imputation method only for IterativeImputer.
    :param min_value: Minimum possible imputed value for IterativeImputer. 
    :param max_value: Maximum possible imputed value for IterativeImputer.
    :param random_state: The seed of the pseudo random number generator to use for IterativeImputer.
    :param n_neighbors: Number of neighboring samples to use for imputation for KNNImputer.
    :param weights: Weight function used in prediction for KNNImputer
                    Valid values: {"uniform", "distance" or "callable" is a user-defined function which accepts an array of distances}.
    :param metric: Distance metric for searching neighbors for KNNImputer; Valid values:{'nan_euclidean','callable'}.


    :rtype: Dataframe with imputed Missing value or selected column with imputed Missing value.


.. note:: Consider non-empty dataset in Dataframe format only 

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("diabetes_train.csv")

    dltk_ai.impute_missing_value(df,['Glucose', 'BloodPressure'],"univariate_imputation",strategy="median")


Treat Outliers
--------------

Function to handle outliers in the data by either removing them or replacing them with a specific value (user specified value or statistic)

.. function:: treat_outliers(dataframe, column, **kwargs):

    :param dataframe: dataframe
    :param column: list of columns/column to treat outliers
    
    `**kwargs`

    :keyword bool remove: True if the outliers should be removed form the data
    :keyword str statistic: min/max/mean/median/quantilevalue
    :keyword int value: bfill/ffill (bfill - backward fill, ffill - front fill)

    :rtype: Dataframe with outliers treated.

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("diabetes_train.csv")

    df_outliers_removed = dltk_ai.treat_outliers(df,['BloodPressure'],remove=True)
    df_outliers_removed.head()


Datatype Conversion
-------------------
Function for Converting datatype of selected column from one datatype to other.

.. function:: convert_dtypes(dataframe,column_datatypes)

    :param dataframe: dataset in the form of pandas Dataframe
    :param column_datatypes: Dictionary as input where keys are column_names and values of dictionary is required datatype for conversion.


    :rtype: Changed Datatype of Dataframe columns

.. note:: Consider dataset in Dataframe only.

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("diabetes_train.csv")

    dltk_ai.convert_dtypes(df,{"Outcome":"int64"})


Feature Scaling
---------------
Function for normalizing the range of independent features of the given dataset.In data processing, it is also known as data normalization
and is generally performed during the data preprocessing step. The range of Feature Scaling can be user-defined or default range provided by 
the particular method.Function for Converting datatype of selected column from one datatype to other.

.. function:: feature_scaling(dataframe,column_names,method,replace=True,feature_range=(0,1),
                              copy=True,clip=False,use_mean=True,use_std=True,
                              use_centering=True, use_scaling=True, 
                              quantile_range=(25.0, 75.0),unit_variance=False,norm='l2')

    :param dataframe: dataset in the form of pandas Dataframe
    :param column_names: List of selected features from dataset for Feature Scaling.
    :param method: Methods from sklearn-library for Feature Scaling; valid values:{"MinMaxScaler","StandardScaler","MaxAbsScaler","RobustScaler",and "Normalizer"}.
    :param replace: Boolean parameter; if True replace original values with Dataframe,otherwise transformed selected column.
    :param feature_range: user-defined range for scaling features given in form of tuple for MinMaxScaler method; by default feature scaled in range of (0,1)
    :param copy: boolean parameter, if true creates copy and then scale the variables otherwise inplaced in same dataframe.
    :param clip: Set to True to clip transformed values of held-out data to provided feature range.
    :param use_mean: Boolean parameter,if True then center the data before scaling for StandardScaler method.
    :param use_std: Boolean parameter,If True then scale the data to unit variance for StandardScaler method.
    :param use_centering: parameter for RobustScaler method
    :param use_scaling: Boolean parameter ,If True then scale the data to interquartile range for RobustScaler method.
    :param quantile_range: Quantile range used for scaling the selected set of features in RobustScaler method.
    :param unit_variance: Boolean parameter,If True then scale data so that normally distributed features have a variance of 1 used in RobustScaler method.
    :param norm: Used to normalize selected columns that are non zero sample for Normalizer method; valid values:{"l1" and "l2"}


    :rtype: Dataframe with scaled features or selected column with scaled features


.. note:: Consider non-empty dataset in Dataframe only.

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("housing_train.csv")

    dltk_ai.feature_scaling(df,['LotArea','2ndFlrSF','GrLivArea'],"MinMaxScaler")



Feature Transformation
----------------------
Function for transforming from one representation to another representation for the selected list of independent features of given dataset.
Different techniques of Feature Transformation can be implemented by checking the skewness of the independent variables. 

.. function:: feature_transformations(dataframe,column_names,transformer,copy=True,
                                      output_distribution='uniform',n_quantiles=1000,
                                      ignore_implicit_zeros=False,subsample=1e5,random_state=None,
                                      method='yeo-johnson',standardize=True,func=None,
                                      inverse_func=None, validate=False, accept_sparse=False,
                                      check_inverse=True, kw_args=None, inv_kw_args=None,
                                      replace=True):

    :param dataframe: dataset in the form of pandas Dataframe
    :param column_names: List of selected features from dataset for Feature Transformation.
    :param transformer: Methods from sklearn-library for Feature Transformation; valid values:{"Quantile Transformer","Power Transformation","Custom Transformation"}
    :param copy: boolean parameter, if true creates copy and then scale the variables otherwise inplaced in same dataframe.
    :param output_distribution: For marginal distribution for the transformed data used in Quantile Transformer; {'uniform', 'normal'}
    :param n_quantiles: Number of quantiles to be computed used in Quantile Transformer where default values is 1000
    :param ignore_implicit_zeros: Boolean parameter used for sparse matrices in Quantile Transformer 
    :param subsample: Maximum number of samples used to estimate the quantiles for computational efficiency in Quantile Transformer; default is 1e5
    :param random_state: Determines random number generation for subsampling and smoothing noise in Quantile Transformer.
    :param method: Used in power transformers; valid values:{‘yeo-johnson’ or ‘box-cox’}
    :param standardize: Boolean parameter when set to True to apply zero-mean, unit-variance normalization to the transformed output for Power Transformer.
    :param func: The callable to use for the transformation in FunctionTransformer. By default its None then func will be the identity function.
    :param inverse_func: The callable to use for the inverse transformation in Custom Transformation.By default its None then func will be the identity function for FunctionTransformer
    :param validate: Boolean parameter, to indicate that the input X array should be checked before calling func for FunctionTransformer.
    :param accept_sparse: Boolean parameter, to indicate that func accepts a sparse matrix as input for FunctionTransformer.
    :param check_inverse: Boolean parameter, to check whether func followed by inverse_func leads to the original inputs for FunctionTransformer.
    :param kw_args: Dictionary of additional keyword arguments to pass to func for FunctionTransformer.
    :param inv_kw_args: Dictionary of additional keyword arguments to pass to inverse_func for FunctionTransformer.
    :param replace: Boolean parameter,if True replace original values with Dataframe,otherwise transformed selected column.


    :rtype: Transformed selected variable in form of dataframe along with columns in the dataset and summary for selected set of columns.

.. note:: Consider non-empty dataset in Dataframe only.

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("housing_train.csv")

    dltk_ai.feature_transformations(df,["MSSubClass",'GarageYrBlt','WoodDeckSF','OpenPorchSF'],transformer="power_transformer")


Feature Creation
----------------

New features created based on existing columns using methods such as binning, one-hot-encoding and group by transform.

.. function:: preprocessor.feature_creation(dataframe, feature_method, binning_column=None, bins=10, 
                            binning_right=True, binning_labels=None,
                     binning_retbins=False, binning_precision=3, binning_include_lowest=False,
                     binning_duplicates='raise', binning_ordered=True, dummies_prefix=None, dummies_prefix_sep='_',
                     dummies_dummy_na=False, dummies_column=None, dummies_sparse=False, dummies_drop_first=False,
                     groupby_column=None, groupby_transform_column=None, groupby_transform_metric='mean')

    
    :param dataframe: dataframe
    :param feature_method: binning/one-hot-encoding/groupby

    binning - bins a numerical variable based on user specified value.
    
    :param binning_column: Dataframe column for binning.
    :param bins: Number of equal width bins. Default - 10
    :param binning_right: Bool, default True. Indicates if the bins should include the right most value.
    :param binning_labels: Array or bool, optional. Labels for the returned bins
    :param bool binning_retbins: Default False. Whether to return the bins or not. Useful when bins is provided as a scalar.
    :param binning_precision: Precision to store and display bins labels.
    :param binning_include_lowest: Whether the first interval should be left-inclusive or not.
    :param binning_duplicates: Raises error if bin edges are not unique. can opt for drop. values = 'raise','drop'.

    one-hot-encoding - Process in the data processing that is applied to categorical data, to convert it into a binary vector representation.

    :param dummies_prefix: List of prefix strings to name the dataframe columns.
    :param dummies_prefix_sep: If appending prefix, separator to use. default '_'
    :param dummies_dummy_na: Add column o indicate NaNs. Default - False.
    :param dummies_sparse: Whether the dummy-encoded columns should be backed by a SparseArray (True) or a regular NumPy array (False).
    :param dummies_drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level.

    groupby - groupby transform returns a self-produced dataframe with transformed values after applying the function specified in its parameter.

    :param groupby_column: list of columns to groupby in the dataframe
    :param groupby_transform_column: column to perform the transform operation on 
    :param groupby_transform_metric: metric to use for transformation - min/max/mean/median. Default - 'mean'

**Example**::

    import dltk_ai
    df = dltk_ai.read_data("diabetes_train.csv")

    # groupby transform
    dltk_ai.feature_creation(df,feature_method='groupby',groupby_column=['Outcome'],groupby_transform_column=['Pregnancies'], groupby_transform_metric='mean')

    # binning
    dltk_ai.feature_creation(df,feature_method='binning',binning_column='Age',bins=20)

    # one-hot-encoding
    dltk_ai.feature_creation(df,feature_method='one-hot-encoding')


Data Transformation
-------------------

Transforms data from one format to another


.. function:: preprocessor.feature_creation(dataframe, transform_method, pivot_index=None, pivot_columns=None, 
                                            pivot_values=None, melt_id_vars=None, melt_value_vars=None, 
                                            crosstab_columns=None, crosstab_rows=None)


    :param dataframe: dataframe
    :param feature_method: pivot/melt/crosstab
    
    pivot - Summarises data in a given dataframe.
    
    :param pivot_index: `str or object or a list of str, optional.` Index column of the new dataframe. 
    :param pivot_columns: `str or object or a list of str`. Columns to make the pivot dataframe.
    :param pivot_values: `str, object or a list of the previous, optional`. Columns for populating pivot dataframe's values.

    melt - Converts a dataframe from wide to long format. Transforms a DataFrame into a format where one or more columns are identifier variables (id_vars), while all other columns, considered measured variables (value_vars), are 'unpivoted' to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’.

    :param melt_id_vars: `tuple, list, or ndarray, optional`. columns to use as identifier variables.
    :param melt_value_vars: `tuple, list, or ndarray, optional`. Columns to unpivot. If not specified uses all columns except the ones specified in melt_id_vars.

    crosstab - Frequency table of factors between 2 or more variables.

    :param crosstab_columns: List of variables for columns in transformed data.
    :param crosstab_rows: List of variables for rows in transformed data.

    :rtype: Reshaped dataframe


**Example**::

    import dltk_ai
    df = dltk_ai.read_data("housing_train.csv")

    # pivot 
    dltk_ai.data_transformation(df,transform_method='pivot',pivot_index='col_B',pivot_columns='col_A')

    # melt
    dltk_ai.data_transformation(df,transform_method='melt',melt_id_vars='col_C',melt_value_vars=['col_A','col_B'])

    # crosstab
    dltk_ai.data_transformation(df,transform_method='crosstab',crosstab_columns='col_A',crosstab_rows='col_B')






