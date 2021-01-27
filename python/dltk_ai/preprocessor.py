import numpy as np
import pandas as pd


def read_data(file_name,header=0,names=None,index_col=None,usecols=None,squeeze=False,mangle_dupe_cols=True,dtype=None,engine=None,converters=None,true_values=None,false_values=None,skiprows=None,skipfooter=0,nrows=None,na_values=None,keep_default_na=True,na_filter=True,verbose=False,parse_dates=False,date_parser=None,thousands=None,comment=None):
    """
    Parameters:
    file_name: complete path of the data file
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
        return pd.read_csv(filepath_or_buffer=file_name,header=header,names=names,index_col=index_col,usecols=usecols, squeeze=squeeze,mangle_dupe_cols=mangle_dupe_cols,dtype=dtype,engine=engine,converters=converters,true_values=true_values,false_values=false_values,skiprows=skiprows,skipfooter=skipfooter,nrows=nrows,na_values=na_values,keep_default_na=keep_default_na,na_filter=na_filter,verbose=verbose,parse_dates=parse_dates,date_parser=date_parser,thousands=thousands,comment=comment)
    elif file_type=='excel':
        if file_name.endswith('.xlsx'):
            engine = 'openpyxl'
        return pd.read_excel(io=file_name,header=header,names=names,index_col=index_col,usecols=usecols, squeeze=squeeze,mangle_dupe_cols=mangle_dupe_cols,dtype=dtype,engine=engine,converters=converters,true_values=true_values,false_values=false_values,skiprows=skiprows,skipfooter=skipfooter,nrows=nrows,na_values=na_values,keep_default_na=keep_default_na,na_filter=na_filter,verbose=verbose,parse_dates=parse_dates,date_parser=date_parser,thousands=thousands,comment=comment)

def read_pickle(filepath_or_buffer, compression='infer'):
    return pd.read_pickle(filepath_or_buffer, compression=compression)


def read_table(filepath_or_buffer, sep='\t', delimiter=None, header='infer', names=None, index_col=None, usecols=None,
               squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
               true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
               na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True,
               parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False,
               cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.',
               lineterminator=None, quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None,
               encoding=None, dialect=None, error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
               low_memory=True, memory_map=False, float_precision=None):
    return pd.read_table(filepath_or_buffer=filepath_or_buffer, sep=sep, delimiter=delimiter, header=header,
                         names=names,
                         index_col=index_col, usecols=usecols, squeeze=squeeze, prefix=prefix,
                         mangle_dupe_cols=mangle_dupe_cols, dtype=dtype, engine=engine, converters=converters,
                         true_values=true_values, false_values=false_values, skipinitialspace=skipinitialspace,
                         skiprows=skiprows, skipfooter=skipfooter, nrows=nrows, na_values=na_values,
                         keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose,
                         skip_blank_lines=skip_blank_lines, parse_dates=parse_dates,
                         infer_datetime_format=infer_datetime_format, keep_date_col=keep_date_col,
                         date_parser=date_parser,
                         dayfirst=dayfirst, cache_dates=cache_dates, iterator=iterator, chunksize=chunksize,
                         compression=compression, thousands=thousands, decimal=decimal, lineterminator=lineterminator,
                         quotechar=quotechar, quoting=quoting, doublequote=doublequote, escapechar=escapechar,
                         comment=comment, encoding=encoding, dialect=dialect, error_bad_lines=error_bad_lines,
                         warn_bad_lines=warn_bad_lines, delim_whitespace=delim_whitespace, low_memory=low_memory,
                         memory_map=memory_map, float_precision=float_precision)


def read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None,
             squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
             true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
             na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True,
             parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False,
             iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None,
             quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None, dialect=None,
             error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False, low_memory=True, memory_map=False,
             float_precision=None):
    return pd.read_csv(filepath_or_buffer=filepath_or_buffer, sep=sep, delimiter=delimiter, header=header, names=names,
                       index_col=index_col, usecols=usecols, squeeze=squeeze, prefix=prefix,
                       mangle_dupe_cols=mangle_dupe_cols, dtype=dtype, engine=engine, converters=converters,
                       true_values=true_values, false_values=false_values, skipinitialspace=skipinitialspace,
                       skiprows=skiprows, skipfooter=skipfooter, nrows=nrows, na_values=na_values,
                       keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose,
                       skip_blank_lines=skip_blank_lines, parse_dates=parse_dates,
                       infer_datetime_format=infer_datetime_format, keep_date_col=keep_date_col,
                       date_parser=date_parser, dayfirst=dayfirst, iterator=iterator, chunksize=chunksize,
                       compression=compression, thousands=thousands, decimal=decimal, lineterminator=lineterminator,
                       quotechar=quotechar, quoting=quoting, doublequote=doublequote, escapechar=escapechar,
                       comment=comment, encoding=encoding, dialect=dialect, error_bad_lines=error_bad_lines,
                       warn_bad_lines=warn_bad_lines, delim_whitespace=delim_whitespace, low_memory=low_memory,
                       memory_map=memory_map, float_precision=float_precision)


def read_fwf(filepath_or_buffer, colspecs='infer', widths=None, infer_nrows=100, **kwds):
    return pd.read_fwf(filepath_or_buffer=filepath_or_buffer, colspecs=colspecs, widths=widths, infer_nrows=infer_nrows,
                       **kwds)


def read_clipboard(sep='\\s+', **kwargs):
    return pd.read_clipboard(sep=sep, **kwargs)


def read_excel(io, sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, dtype=None,
               engine=None, converters=None, true_values=None, false_values=None, skiprows=None, nrows=None,
               na_values=None, keep_default_na=True, na_filter=True, verbose=False, parse_dates=False, date_parser=None,
               thousands=None, comment=None, skipfooter=0, convert_float=True, mangle_dupe_cols=True):
    return pd.read_excel(io=io, sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols,
                         squeeze=squeeze, dtype=dtype, engine=engine, converters=converters, date_parser=date_parser,
                         true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows,
                         na_values=na_values, keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose,
                         parse_dates=parse_dates, thousands=thousands, comment=comment, skipfooter=skipfooter,
                         convert_float=convert_float, mangle_dupe_cols=mangle_dupe_cols)


def ExcelFile_parse(sheet_name=0, header=0, names=None, index_col=None, usecols=None, squeeze=False, converters=None,
                    true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, parse_dates=False,
                    date_parser=None, thousands=None, comment=None, skipfooter=0, convert_float=True,
                    mangle_dupe_cols=True, **kwds):
    return pd.ExcelFile.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols,
                              squeeze=squeeze, converters=converters, true_values=true_values,
                              false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values,
                              parse_dates=parse_dates, date_parser=date_parser, thousands=thousands, comment=comment,
                              skipfooter=skipfooter, convert_float=convert_float, mangle_dupe_cols=mangle_dupe_cols,
                              **kwds)


def ExcelWriter(path, engine=None, **kwargs):
    return pd.ExcelWriter(path=path, engine=engine, **kwargs)


def read_json(path_or_buf=None, orient=None, typ="frame", dtype=None, convert_axes=None, convert_dates=True,
              keep_default_dates=True, numpy=False, precise_float=False, date_unit=None, encoding=None, lines=False,
              chunksize=None, compression="infer"):
    return pd.read_json(path_or_buf=path_or_buf, orient=orient, typ=typ, dtype=dtype, convert_axes=convert_axes,
                        convert_dates=convert_dates, keep_default_dates=keep_default_dates, numpy=numpy,
                        precise_float=precise_float, date_unit=date_unit, encoding=encoding, lines=lines,
                        chunksize=chunksize, compression=compression)


def json_normalize(data, record_path=None, meta=None, meta_prefix=None, record_prefix=None, errors='raise', sep='.',
                   max_level=None):
    return pd.json_normalize(data=data, record_path=record_path, meta=meta, meta_prefix=meta_prefix,
                             record_prefix=record_prefix, errors=errors, sep=sep, max_level=max_level)


def read_html(io, match=".+", flavor=None, header=None, index_col=None, skiprows=None, attrs=None, parse_dates=False,
              thousands=",", encoding=None, decimal=".", converters=None, na_values=None, keep_default_na=True,
              displayed_only=True):
    return pd.read_html(io=io, match=match, flavor=flavor, header=header, index_col=index_col, skiprows=skiprows,
                        attrs=attrs, parse_dates=parse_dates, thousands=thousands, encoding=encoding, decimal=decimal,
                        converters=converters, na_values=na_values, keep_default_na=keep_default_na,
                        displayed_only=displayed_only)


def read_hdf(path_or_buf, key=None, mode='r', errors='strict', where=None, start=None, stop=None, columns=None,
             iterator=False, chunksize=None, **kwargs):
    return pd.read_hdf(path_or_buf=path_or_buf, key=key, mode=mode, errors=errors, where=where, start=start, stop=stop,
                       columns=columns, iterator=iterator, chunksize=chunksize, **kwargs)


def read_feather(path, columns=None, use_threads=True):
    return pd.read_feather(path=path, columns=columns, use_threads=use_threads)


def read_parquet(path, engine='auto', columns=None, **kwargs):
    return pd.read_parquet(path=path, engine=engine, columns=columns, **kwargs)


def read_orc(path, columns=None, **kwargs):
    return pd.read_orc(path=path, columns=columns, **kwargs)


def read_sas(filepath_or_buffer, format=None, index=None, encoding=None, chunksize=None, iterator=False, ):
    return pd.read_sas(filepath_or_buffer=filepath_or_buffer, format=format, index=index, encoding=encoding,
                       chunksize=chunksize, iterator=iterator, )


def read_spss(path, usecols=None, convert_categoricals=True):
    return pd.read_spss(path=path, usecols=usecols, convert_categoricals=convert_categoricals)


def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True, parse_dates=None, columns=None,
                   chunksize=None, ):
    return pd.read_sql_table(table_name=table_name, con=con, schema=schema, index_col=index_col,
                             coerce_float=coerce_float,
                             parse_dates=parse_dates, columns=columns, chunksize=chunksize, )


def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None, ):
    return pd.read_sql_query(sql=sql, con=con, index_col=index_col, coerce_float=coerce_float, params=params,
                             parse_dates=parse_dates, chunksize=chunksize, )


def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, columns=None,
             chunksize=None, ):
    return pd.read_sql(sql=sql, con=con, index_col=index_col, coerce_float=coerce_float, params=params,
                       parse_dates=parse_dates, columns=columns, chunksize=chunksize, )


def read_gbq(query, project_id=None, index_col=None, col_order=None, reauth=False, auth_local_webserver=False,
             dialect=None, location=None, configuration=None, credentials=None, use_bqstorage_api=None,
             max_results=None, private_key=None, verbose=None, progress_bar_type=None):
    return pd.read_gdq(query=query, project_id=project_id, index_col=index_col, col_order=col_order, reauth=reauth,
                       auth_local_webserver=auth_local_webserver, dialect=dialect, location=location,
                       configuration=configuration, credentials=credentials, use_bqstorage_api=use_bqstorage_api,
                       max_results=max_results, private_key=private_key, verbose=verbose,
                       progress_bar_type=progress_bar_type)


def read_stata(filepath_or_buffer, convert_dates=True, convert_categoricals=True, index_col=None, convert_missing=False,
               preserve_dtypes=True, columns=None, order_categoricals=True, chunksize=None, iterator=False):
    return pd.read_stata(filepath_or_buffer=filepath_or_buffer, convert_dates=convert_dates,
                         convert_categoricals=convert_categoricals, index_col=index_col,
                         convert_missing=convert_missing, preserve_dtypes=preserve_dtypes, columns=columns,
                         order_categoricals=order_categoricals, chunksize=chunksize, iterator=iterator)


def data_profile(dataframe):
    """
    Parameters:
    dataframe : dataframe

    Returns - Data Dictionary
    """
    # get a list of columns
    columns = list(dataframe)  # list of columns
    # get list of unique values in each column
    unique_values = list(dataframe.nunique().values)
    # get list of missing values in each column
    missing_values = list(dataframe.isnull().sum().values)
    # get list of missing value % in each column
    missing_value_percentage = list((dataframe.isnull().sum() * 100) / len(dataframe))
    # get datatype of each column
    data_types = list(dataframe.dtypes.values)

    param = []
    value_list = []
    # for each column, append the param in pandas describe function and the value associated with it - include='all' will give the stats of categorical variables also
    for i in range(len(dataframe.describe(include='all').index)):
        param.append(dataframe.describe(include='all').index[i])
        value_list.append(list(dataframe.describe(include='all').iloc[i].values))

    # outliers - higher/lower
    # calculate outliers for numerical columns 
    higher_outlier = []
    lower_outlier = []
    for i in list(dataframe):
        # check if the datatype of column id int64
        if dataframe[i].dtypes == 'int64' or dataframe[i].dtypes == 'float64':
            higher = dataframe[i].quantile(0.75) + 1.5 * (dataframe[i].quantile(0.75) - dataframe[i].quantile(0.25))
            lower = dataframe[i].quantile(0.25) - 1.5 * (dataframe[i].quantile(0.75) - dataframe[i].quantile(0.25))
            higher_outlier.append(higher)
            lower_outlier.append(lower)
        # if not int64 append null values
        else:
            higher_outlier.append(np.nan)
            lower_outlier.append(np.nan)
    # create a dataframe with all the above metrics related to each columna dn return the dataframe
    describe_dict = dict(zip(param, value_list))
    properties_dict = {'columns': columns, 'data_types': data_types, 'unique_values': unique_values,
                       'missing_values': missing_values, 'missing_value_percentage': missing_value_percentage}
    outlier_dict = {"higher": higher_outlier, "lower": lower_outlier}

    describe_dict.update(properties_dict)
    describe_dict.update(outlier_dict)

    # put above things in a dataframe
    eda_df = pd.DataFrame(describe_dict)
    return eda_df


def treat_missing_data(dataframe, column, **kwargs):
    """
    Parameters:
    dataframe
    column: list of columns/column to treat missing data
    **kwargs: Default - statistic - mean
        give any on of the below parameters - either statistic/value/fill_method
        statistic: min/max/mean/median/quantilevalue
        value: user specified value
        fill_method: bfill/ffill (bfill - backward fill, ffill - front fill)
    """
    # if the parameter in kwargs is not given, take default as statistic-mean
    if len(kwargs) == 0:
        kwargs['statistic'] = 'mean'
    dataframe_duplicate = dataframe.copy()
    # if more than one kwargs is given raise error
    if len(kwargs.keys()) > 1:
        raise ValueError('please select any one argument value/statistic/fill_method')
    else:
        # if params in kwargs are not valid, raise error
        if list(kwargs.keys())[0] not in ['value', 'statistic', 'fill_method']:
            raise ValueError('please choose a valid argument')
        else:
            new_columns = []
            for i in column:
                # if the variable is categorical, fill missing values with mode
                if dataframe_duplicate[i].dtypes not in ['int64', 'float64']:
                    kwargs['statistic'] = 'mode'
                else:
                    pass
                # variable for given kwarg - value/statistic/fill_method
                arg = list(kwargs.keys())[0]
                if arg == 'value':
                    # if value - fill missing values with the specified value
                    new_columns.append(dataframe_duplicate[i].fillna(kwargs[arg]))

                elif arg == 'statistic':
                    # if statistic - fill with specified statistic
                    statistic_value = dataframe_duplicate[i].mode() if kwargs[arg] == 'mode' else dataframe_duplicate[
                        i].max() if kwargs[arg] == 'max' else dataframe_duplicate[i].min() if kwargs[arg] == 'min' else \
                        dataframe_duplicate[i].mean() if kwargs[arg] == 'mean' else dataframe_duplicate[i].median() if \
                            kwargs[arg] == 'median' else dataframe_duplicate[i].quantile(kwargs[arg]) if (
                                type(kwargs[arg]) == int or type(kwargs[arg]) == float) else 0
                    column_array = dataframe_duplicate[i]
                    new_columns.append(column_array.fillna(statistic_value))
                # if arg is fill_method, front fill or back fill
                elif arg == 'fill_method':
                    new_columns.append(dataframe_duplicate[i].fillna(method=kwargs[arg]))
                else:
                    pass
    return new_columns


def treat_outliers(dataframe, column, **kwargs):
    """
    Parameters
    dataframe
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
    # print(remove)
    upper = kwargs.get('upper', True)
    # print(upper)
    lower = kwargs.get('lower', True)
    # print(lower)
    new_column = []
    for i in column:

        higher_outlier = dataframe_duplicate[i].quantile(0.75) + 1.5 * (
                dataframe_duplicate[i].quantile(0.75) - dataframe_duplicate[i].quantile(0.25))
        # print(higher_outlier)
        lower_outlier = dataframe_duplicate[i].quantile(0.25) - 1.5 * (
                dataframe_duplicate[i].quantile(0.75) - dataframe_duplicate[i].quantile(0.25))
        # print(lower_outlier)
        if remove:
            # print('in remove')
            # print(kwargs['remove'])
            if kwargs['remove'] not in [True, False]: raise ValueError('remove should be either True/False')
            # print(dataframe_duplicate[i])
            if upper:
                dataframe_duplicate = dataframe_duplicate[dataframe_duplicate[i] <= higher_outlier]
            if lower:
                dataframe_duplicate = dataframe_duplicate[dataframe_duplicate[i] >= lower_outlier]
        else:
            if 'value' in kwargs:
                if type(kwargs['value']) != int: raise ValueError('value should be a interger')
                value = kwargs['value']
            elif 'statistic' in kwargs:
                if kwargs['statistic'] in ['min', 'max', 'mean', 'median'] or type(kwargs['statistic']) == int:
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
                    kwargs['statistic']) == int else 0
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
