"""Financial utility functions."""
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web

import pytrends
from pytrends.request import TrendReq

from fastai.structured import add_datepart

DATA_PATH = "data"


def create_ml_features_df(data_frame) :
    """Create a DataFrame with non-ticker specific features for machine
    learning.
    
    Features added as new columns:
        - 'Days Since Trading': Days since last trading day
        - 'Days Until Trading': Days until next trading day
        - Various date part columns from fastai.structured.add_datepart

    Args:
        data_frame: DataFrame to add features to.

    Returns:
        A new DataFrame with features added to data_frame. 
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    data = create_days_since_valid_date(data_frame, 'Days Since Trading')

    # Get the days until next day of trading. Reverse sort data, call same 
    # function as above, then resort in normal order.
    data.sort_index(ascending=False, inplace=True)
    data = create_days_since_valid_date(data, 'Days Until Trading')
    data.sort_index(ascending=True, inplace=True)

    # Add separate columns for various date parts (month, day, etc.)
    data = data.reset_index()
    add_datepart(data, 'Date', drop=False)
    data = data.set_index('Date')
    
    return data


def create_ml_ticker_features_df(ticker) :
    """Create a DataFrame with added ticker-specific features useful for machine
    learning.
    
    Args:
        data_frame: DataFrame to add features to.

    Returns:
        A new DataFrame with features added to data_frame.     
    """
    data = create_price_data(ticker)
    
    #Get the price ranges: (High - Low), (Open - Close)
    data['Daily Range'] = data['High'] - data['Low']
    data['Daily Gain'] = data['Close'] - data['Open']
    data['Close Higher than Open'] = data['Close'] > data['Open']
    data['Close Lower than Open'] = data['Close'] < data['Open']

    data['High was Open'] = abs(data['High'] - data['Open']) < 0.001
    data['High was Close'] = abs(data['High'] - data['Close']) < 0.001
    data['Low was Open'] = abs(data['Low'] - data['Open']) < 0.001
    data['Low was Close'] = abs(data['Low'] - data['Close']) < 0.001

    data['Closed Higher than Prev Close'] = data['Close'].diff() > 0
    data['Closed Lower than Prev Close'] = data['Close'].diff() < 0

    # SMAs of Adj Close
    sma = create_sma_df(data, 'Adj Close')
    del sma['Adj Close']
    data = pd.DataFrame.join(data, sma)

    sma_pct_diff = create_sma_pct_diff_df(data, 'Adj Close')
    del sma_pct_diff['Adj Close']
    data = pd.DataFrame.join(data, sma_pct_diff)

    # SMAs of Volume
    sma = create_sma_df(data, 'Volume')
    del sma['Volume']
    data = pd.DataFrame.join(data, sma)

    sma_pct_diff = create_sma_pct_diff_df(data, 'Volume')
    del sma_pct_diff['Volume']
    data = pd.DataFrame.join(data, sma_pct_diff)

    return data


def add_prefix_to_column_names(data_frame, prefix) :
    """Add prefix to all column names in data_frame.""" 
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    new_columns = []
    for c in data_frame.columns :
        new_columns.append(f'{prefix} {c}')
        data_frame.columns = new_columns
        data_frame.head().T


def get_df_start_date(data_frame) :
    assert (isinstance(data_frame, pd.DataFrame)), \
        "data_frame must be pandas.DataFrame object"

    return min(data_frame.index).date()


def get_df_end_date(data_frame) :
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    return max(data_frame.index).date()


def create_price_data(ticker) :
    """Get a DataFrame with the price data for ticker. Rows with NA values will 
    be removed from data."""
    # Defualt to yahoo for now
    data = create_price_data_yahoo(ticker).dropna()
    
    start_date = get_df_start_date(data)
    end_date = get_df_end_date(data)
    print(f'Loaded data for {ticker}: {start_date} to {end_date}.')
    
    return data

def update_price_data_yahoo(price_data, ticker) :
    """Updates price data to include most recent data available from Yahoo.
    """
    assert isinstance(price_data, pd.DataFrame), \
        "price_data must be pandas.DataFrame object"

    # try to update from next day after end of data
    price_data_end_date = get_df_end_date(price_data) 
    req_start_date =  price_data_end_date + dt.timedelta(1)
    # ... until yesterday
    req_end_date = dt.date.today() - dt.timedelta(1)
    
    # Might not need to do update
    if (req_end_date <= req_start_date) :
        return price_data

    price_data_path = f'{DATA_PATH}/yahoo/{ticker}.csv'
    
    # default to original data if update fails
    updated_data = price_data
    
    try :
        updated_data = web.DataReader(ticker, 'yahoo', req_start_date, 
            req_end_date)
        actual_end_date = get_df_end_date(updated_data)

        # just return original price_data if no update needed
        if (actual_end_date <= price_data_end_date) :
            print(f'No updates to {ticker} found from yahoo.')
            return price_data
        else :
            updated_data = price_data.append(updated_data)
            print(f'Updated data for {ticker} from yahoo: '
                f'{req_start_date} to {actual_end_date}')
            updated_data.to_csv(price_data_path)
    except :
        print(f'Could not load updates for {ticker} from yahoo. '
            'Using cached data.')

    return updated_data        

def create_price_data_yahoo(ticker) :
    """Get raw ticker price data from yahoo. No data cleaning is performed on 
    data."""
    start = dt.datetime(1970, 1, 1)
    end = dt.date.today() - dt.timedelta(1)
    
    price_data_path = f'{DATA_PATH}/yahoo/{ticker}.csv'
    
    # Load local copy
    try :
        # read local copy
        price_data = pd.read_csv(price_data_path, parse_dates=True, index_col=0)
        # try to update
        price_data = update_price_data_yahoo(price_data, ticker)
        #price_data.set_index('Date', inplace=True)
        # FIXME check if up to date, if not update and save
    except :
        print(f'Could not read file: {price_data_path}. ' 
            f'Downloading data for "{ticker}" from yahoo.com...')
        price_data = web.DataReader(ticker, 'yahoo', start, end)
        price_data.to_csv(price_data_path)
    
    return price_data


def add_ema_column(data_frame, col_name, num_days) :
    """Add a column with Exponential Moving Average (EMA) data to a DataFrame.

    Args:
        data_frame: DataFrame containing data of interest.
        col_name: String containing column of interest. Column is assumed to 
            contain daily data.
        num_days: Number of days (span) to use for calculating EMA.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
 
    data_frame[f'{col_name} EMA{num_days}'] = \
        data_frame[col_name].ewm(span=num_days).mean()


def add_sma_column(data_frame, col_name, num_days) :
    """Add a column with Simple Moving Average (SMA) data to a DataFrame."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    data_frame[f'{col_name} SMA{num_days}'] = \
        data_frame[col_name].rolling(window=num_days).mean()


def get_pct_diff(measured, expected) :
    """Calcuate the percent difference between measured and expected values."""
    return 100*(measured - expected)/expected


def add_sma_pct_diff_column(data_frame, col_name, sma_period) :
    """Add a column with the percent difference between data and the Simple 
    Moving Average of that data."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    sma = data_frame[col_name].rolling(window=sma_period).mean()
    data_frame[f'pct diff {col_name} SMA{sma_period}'] = \
        get_pct_diff(data_frame[col_name], sma)


def add_ema_pct_diff_column(data_frame, col_name, ema_period) :
    """Add a column with the percent difference between base values and the 
    Exponential Moving Average of those values."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    ema = data_frame[col_name].ewm(span=ema_period).mean()
    data_frame[f'pct diff {col_name} EMA{ema_period}'] = \
        get_pct_diff(data_frame[col_name], ema)


def create_ema_pct_diff_df(data_frame, col_name, 
        ema_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Create a DataFrame containing base values plus several columns of percent 
    difference between those values and the Exponential Moving Averages (EMAs) 
    of those values for various periods.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    ema_pct_diff_df = pd.DataFrame(data_frame[col_name])
    for d in ema_periods_list :
        add_ema_pct_diff_column(ema_pct_diff_df, col_name, d)

    return ema_pct_diff_df



def create_sma_pct_diff_df(data_frame, col_name, 
        sma_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Create a DataFrame containing base values plus several columns of percent 
    difference between those values and the Simple Moving Averages (EMAs) of 
    those values for various periods.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    sma_pct_diff_df = pd.DataFrame(data_frame[col_name])
    for d in sma_periods_list :
        add_sma_pct_diff_column(sma_pct_diff_df, col_name, d)

    return sma_pct_diff_df


def create_ema_df(data_frame, col_name, 
        ema_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Create a DataFrame containing base values plus several columns of the 
    Exponential Moving Averages (EMAs) of those values for various periods.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    ema_df = pd.DataFrame(data_frame[col_name])
    for d in ema_periods_list :
        add_ema_column(ema_df, col_name, d)

    return ema_df


def create_sma_df(data_frame, col_name,
        sma_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Create a DataFrame containing base values plus several columns of the 
    Simple Moving Averages (EMAs) of those values for various periods.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    sma_df = pd.DataFrame(data_frame[col_name])
    for d in sma_periods_list :
        add_sma_column(sma_df, col_name, d)

    return sma_df


def create_macd_df(data_frame, col_name) :
    """Create a DataFrame containing base values plus the MACD components:
    Fast EMA, Slow EMA, MACD, Signal, Histogram."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    FAST = 'Fast EMA'
    SLOW = 'Slow EMA'
    MACD = 'MACD'
    SIGNAL = 'Signal'
    HIST = 'Histogram'

    macd_df = pd.DataFrame(data_frame[col_name])
    macd_df[FAST] = macd_df[col_name].ewm(span=12).mean()
    macd_df[SLOW] = macd_df[col_name].ewm(span=26).mean()
    macd_df[MACD] = macd_df[FAST] - macd_df[SLOW]
    macd_df[SIGNAL] = macd_df[MACD].ewm(span=9).mean()
    macd_df[HIST] = macd_df[MACD] - macd_df[SIGNAL]

    return macd_df


def create_google_trends_sma_pct_diff_df(data_frame, search, 
        sma_periods_list=[3,5,10,20,50,100,200]) :
    """Create a DataFrame containing the percent difference of google search 
    trends with respect to Simple Moving Averages of trend data of various 
    periods."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    
    trend = create_google_trends_df(data_frame, search)
    trend = create_sma_pct_diff_df(trend, search, sma_periods_list)
    del trend[search]
    return trend

def create_days_since_valid_date(data_frame, new_col_name) :
    """Copy a DataFrame and add a column containing the number of days elapsed
    since the last valid date. 
    
    Args:
        data_frame: DataFrame with a DatetimeIndex.
        new_col_name: String with the name of the new column containg the days
            elapsed since last valid date.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    assert isinstance(data_frame.index, pd.DatetimeIndex), \
        "data_frame must use a DatetimeIndex"

    elapsed_df = pd.DataFrame(data_frame)
    day_length = np.timedelta64(1, 'D')
    last_date = data_frame.index.values[0]
    days_elapsed = []

    for d in data_frame.index.values :
        days_since_last = (abs(d-last_date).astype('timedelta64[D]') / 
            day_length).astype(int)
        days_elapsed.append(days_since_last)
        last_date = d

    elapsed_df[new_col_name] = days_elapsed
    return elapsed_df


def create_days_since_valid_value(data_frame, col_name, new_name_prefix):
    """Copy a DataFrame and add a column which gives the days elapsed since the
    column of interest had a valid value.

    Args:
        data_frame: DataFrame of interest, must have DatetimeIndex.
        col_name: String with name of column of interest in data_frame.
        new_name_prefix: String with a prefix that will be added to col_name to 
            create new column. Newly added column will have the name 
            'f{new_name_prefix} {col_name}'.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    assert isinstance(data_frame.index, pd.DatetimeIndex), \
        "data_frame must use a DatetimeIndex"

    elapsed_df = pd.DataFrame(data_frame)
    day_length = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    days_elapsed = []

    for v,d in zip(data_frame[col_name].values, data_frame.index.values):
        if not pd.isna(v) :
            last_date = d
        days_since_valid = ((d-last_date).astype('timedelta64[D]') /
            day_length).astype(int)
        days_elapsed.append(days_since_valid)
        
    elapsed_df[f'{new_name_prefix} {col_name}'] = days_elapsed
    return elapsed_df


def create_google_trends_df(data_frame, search):
    """Create a DataFrame with google trends for a search.""" 
    # create data_range
    date_range = \
        [f'{get_df_start_date(data_frame)} {get_df_end_date(data_frame)}']
    
    # Set up the trend fetching object
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [search]

    try:
        # Create the search object
        pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='',
            gprop='news')
        
        # Retrieve the interest over time
        trends = pytrends.interest_over_time()
        #trends = pd.DataFrame(trends_series, search)

    except Exception as e:
        print('\nGoogle Search Trend retrieval failed.')
        print(e)
        return

    # Upsample the data to daily
    trends = trends.resample('D').mean()
    # add column indicating how long since trend updated
    trends = create_days_since_valid_value(trends, search, 'Days since updated')
    # clean up na values from upsample
    trends = pd.DataFrame.fillna(trends, method='ffill')
    
    return trends


def create_fitted_line_df(data_frame, col_name, fitted_col_name) :
    '''Create a DataFrame with base values of interest plus a column with a best
    fit line for those values.

    Args:
        data_frame: DataFrame of interest, must use a DatetimeIndex.
        col_name: String with name of the column of interest. Any NA values in
            col_name will be filled in with predicted values in fitted_col_name.
        fitted_col_name: String with the name of the new column that contains 
            the best fit line values. 
    Returns:
        A DataFrame containing a column with the values of interest plus a 
        column with the best fit line values.
    '''
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    assert isinstance(data_frame.index, pd.DatetimeIndex), \
        "data_frame must use a DatetimeIndex"
    assert (col_name is not None)
    assert (fitted_col_name is not None)
    
    df = pd.DataFrame(data_frame[col_name])
    
    # resample to daily to get even spacing -- this introduces NA values, but 
    # drop them later after getting a nice index for fitting
    df = df.resample('D').mean()

    # reset index twice to get an 'index' column, which is nicely numbered for
    # use in fitting -- dates don't work so well 
    df = df.reset_index().reset_index()

    # reset index to Date
    df = df.set_index('Date')
    
    # Now that we have an evenly spaced and nicely numbered index, make a 
    # working copy and drop NAs
    fit_df = df.dropna()

    # ... and get x and y values for fitting
    x = fit_df['index'].astype('float')
    y = fit_df[col_name]
    
    # Fit the values, and get the prediction function
    model = np.polyfit(x, y, deg=1)
    predict = np.poly1d(model)
    
    # get predicted values for original data_frame
    fit_vals = predict(df['index'])
    df[fitted_col_name] = fit_vals
    
    #return_df = pd.DataFrame(data_frame[col_name])
    #return_df = return_df.join(df, how='left')
    
    # delete the index values and the original data
    del df['index']
    del df[col_name]
    
    # now we'll join back to the original data_frame to eliminate the extra 
    # data added by the resampling
    df2 = pd.DataFrame(data_frame[col_name])
    df = df2.join(df, how='inner')
    del df[col_name]
        
    return df