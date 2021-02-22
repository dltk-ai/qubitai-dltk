************
Preprocessor
************

Read Data
---------

Function for reading data from csv or excel to dataframe.

.. function:: preprocessor.read_data(file_name,header=0,names=None,index_col=None,usecols=None,
                                    squeeze=False,mangle_dupe_cols=True,dtype=None,engine=None,
                                    converters=None,true_values=None,false_values=None,skiprows=None,
                                    skipfooter=0,nrows=None,na_values=None,keep_default_na=True,
                                    na_filter=True,verbose=False,parse_dates=False,date_parser=None,
                                    thousands=None,comment=None)


    :param data_path: path for data file - csv or excel
    :param file_name: complete path of the data file
    :param header: Row number(s) to use as the column names. default 0 (first row)
    :param names: List of column names to use. array-like, optional.
    :param index_col: Column(s) to use as the row labels of the DataFrame. int, str, sequence of int / str, or False, default None.
    :param usecols: a list of subset of columns
    :param squeeze: If the parsed data only contains one column then return a Series. bool, default False
    :param mangle_dupe_cols: Duplicate columns will be specified with extension such as 1,2..etc. bool, default True
    :param dtype: specify datatype of particular column while reading eg: {‘a’: ‘Int64’}
    :param engine: Parser engine to use. {‘c’, ‘python’}, optional
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
    :param parse_dates: bool or list of int or names or list of lists or dict, default False
    :param date_parser: Function to use for converting a sequence of string columns to an array of datetime instances. function, optional.
    :param thousands: Thousands separator. str, optional.
    :param comment: Indicates remainder of line should not be parsed. str, optional.

.. note:: Supports csv, xls & xlsx formats only.

**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    df = preprocessor.read_data(../data.csv)
    print(df)


Data Profile
------------
Information describing the statistics of variables in the data. 

.. function:: preprocessor.data_profile(dataframe)

   :param dataframe: dataframe
   :rtype: A dataframe with data dictionary

**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    df_profile = preprocessor.data_profile(df)
    print(df_profile)



Missing Values
---------------

Function to handle missing values by replacing them with a statistic or user specified value or methods such as back fill and front fill.

.. function:: preprocessor.treat_missing_data(dataframe, column, **kwargs):

    :param dataframe: dataframe
    :param column: list of columns/column to treat missing data

    `**kwargs`

    :keyword str statistic: min/max/mean/median/quantilevalue
    :keyword int value: user specified value
    :keyword str fill_method: bfill/ffill (bfill - backward fill, ffill - front fill)
    
    :rtype: a list of series of the columns where the missing values are imputed

.. note::
    Only one param can be used in statistic, value & fill_method.
    For categorical variables, most repeated value is default value for filling missing values.

**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    handling_missing_data = preprocessor.treat_missing_data(df, ['col_A','col_B'], statistic = 'min')
    print(handling_missing_data)


Treat Outliers
--------------

Function to handle outliers in the data by either removing them or replacing them with a specific value (user specified value or statistic)

.. function:: preprocessor.treat_outliers(dataframe, column, **kwargs):

    :param dataframe: dataframe
    :param column: list of columns/column to treat outliers
    `**kwargs`

    :keyword bool remove: True if the outliers should be removed form the data
    :keyword str statistic: min/max/mean/median/quantilevalue
    :keyword int value: bfill/ffill (bfill - backward fill, ffill - front fill)

    :rtype: Dataframe with outliers treated.

**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    df_outliers_removed = preprocessor.treat_outliers(player_df, ['col_D','col_E'], remove = True)
    df_outliers_removed.head()


Feature Creation
----------------

Creates new features based on existing columns using methods such as binning, one-hot-encoding & groupby transform.

.. function:: preprocessor.feature_creation(dataframe, feature_method, binning_column=None, bins=10, 
                            binning_right=True, binning_labels=None,
                     binning_retbins=False, binning_precision=3, binning_include_lowest=False,
                     binning_duplicates='raise', binning_ordered=True, dummies_prefix=None, dummies_prefix_sep='_',
                     dummies_dummy_na=False, dummies_column=None, dummies_sparse=False, dummies_drop_first=False,
                     groupby_column=None, groupby_transform_column=None, groupby_transform_metric='mean')

    
    :param dataframe: dataframe
    :param feature_method: binning/one-hot-encoding/groupby

    binning - bins a numerical variable based on user specified value.
    
    :param binning_column: dataframe column for binning.
    :param bins: Number of equal width bins. Default - 10
    :param binning_right: bool, default True. Indicates if the bins should include the right most value.
    :param binning_labels: array or bool, optional. Labels for the returned bins
    :param bool binning_retbins: Default False. Whether to return the bins or not. Useful when bins is provided as a scalar.
    :param binning_precision: precision to store and display bins labels.
    :param binning_include_lowest: Whether the first interval should be left-inclusive or not.
    :param binning_duplicates: raises error if bin edges are not unique. can opt for drop. values = 'raise','drop'.

    one-hot-encoding - Process in the data processing that is applied to categorical data, to convert it into a binary vector representation.

    :param dummies_prefix: list of prefix strings to name the dataframe columns.
    :param dummies_prefix_sep: if appending prefix, separator to use. default '_'
    :param dummies_dummy_na: Add columnt o indicate NaNs. Default - False.
    :param dummies_sparse: Whether the dummy-encoded columns should be backed by a SparseArray (True) or a regular NumPy array (False).
    :param dummies_drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level.

    groupby - groupby transform returns a self-produced dataframe with transformed values after applying the function specified in its parameter.

    :param groupby_column: list of columns to groupby in the dataframe
    :param groupby_transform_column: column to perform the transform operation on 
    :param groupby_transform_metric: metric to use for transformation - min/max/mean/median. Default - 'mean'

**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    # groupby transform
    preprocessor.feature_creation(df,feature_method='groupby',groupby_column=['col_A','col_B'],groupby_transform_column=['col_C'])

    # binning
    preprocessor.feature_creation(df,feature_method='binning',binning_column='col_G',bins=20)

    # one-hot-encoding
    preprocessor.feature_creation(df,feature_method='one-hot-encoding')


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

    :param crosstab_columns: list of variables for columns in transformed data. 
    :param crosstab_rows: list of variables for rows in transformed data.

    :rtype: reshaped dataframe


**Example**::

    import dltk_ai
    from dltk_ai import preprocessor

    # pivot 
    preprocessor.data_transformation(df,transform_method='pivot',pivot_index='col_B',pivot_columns='col_A')

    # melt
    preprocessor.data_transformation(df,transform_method='melt',melt_id_vars='col_C',melt_value_vars=['col_A','col_B'])

    # crosstab
    preprocessor.data_transformation(df,transform_method='crosstab',crosstab_columns='col_A',crosstab_rows='col_B')





