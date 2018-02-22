"""Financial utility functions."""
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web

import pytrends
from pytrends.request import TrendReq

from fastai.structured import add_datepart

DATA_PATH = "data"

def create_dl_features(data_frame) :
    """Return a DataFrame with added features useful for deep learning (dl) that aren't ticker
    specific. For example, date parts, and days since last trading day."""
    data = add_days_since_valid_date(data_frame, 'Days Since Trading')

    # Get the days until next day of trading. Reverse sort data, call same function as above, 
    # then resort in normal order.
    data.sort_index(ascending=False, inplace=True)
    data = add_days_since_valid_date(data, 'Days Until Trading')
    data.sort_index(ascending=True, inplace=True)

    # Add separate columns for various date parts (month, day, etc.)
    data = data.reset_index()
    add_datepart(data, 'Date', drop=False)
    data = data.set_index('Date')
    
    return data


def create_dl_ticker_features(ticker) :
    """Return a DataFrame with ticker data plus created features useful for deep learning (dl)."""
    data = get_price_data(ticker)
    
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
    sma = get_sma_df(data, 'Adj Close')
    del sma['Adj Close']
    data = pd.DataFrame.join(data, sma)

    sma_pct_diff = get_sma_pct_diff_df(data, 'Adj Close')
    del sma_pct_diff['Adj Close']
    data = pd.DataFrame.join(data, sma_pct_diff)

    # SMAs of Volume
    sma = get_sma_df(data, 'Volume')
    del sma['Volume']
    data = pd.DataFrame.join(data, sma)

    sma_pct_diff = get_sma_pct_diff_df(data, 'Volume')
    del sma_pct_diff['Volume']
    data = pd.DataFrame.join(data, sma_pct_diff)

    return data


def add_prefix_to_column_names(data_frame, prefix) :
    """Add prefix to all column names in data_frame.""" 
    new_columns = []
    for c in data.columns :
        new_columns.append(f'{TICKER} {c}')
        data.columns = new_columns
        data.head().T


def get_df_start_date(data_frame) :
    return min(data_frame.index).date()


def get_df_end_date(data_frame) :
    return max(data_frame.index).date()


def get_price_data(ticker) :
    """Get price data for ticker. Rows with NA values will be removed from data."""
    # Defualt to yahoo for now
    data = get_price_data_yahoo(ticker).dropna()
    start_date = get_df_start_date(data)
    end_date = get_df_end_date(data)


    print(f'Loaded data for {ticker}: {start_date} to {end_date}.')
    return data

def update_price_data_yahoo(price_data, ticker) :
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
        updated_data = web.DataReader(ticker, 'yahoo', req_start_date, req_end_date)
        actual_end_date = get_df_end_date(updated_data)

        # just return original price_data if no update needed
        if (actual_end_date <= price_data_end_date) :
            print(f'No updates to {ticker} found from yahoo.')
            return price_data
        else :
            updated_data = price_data.append(updated_data)
            print(f'Updated data for {ticker} from yahoo: {req_start_date} to {actual_end_date}')
            updated_data.to_csv(price_data_path)
    except :
        print(f'Could not load updates for {ticker} from yahoo. Using cached data.')

    return updated_data        

def get_price_data_yahoo(ticker) :
    """Get raw ticker price data from yahoo. No data cleaning is performed on data."""
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
        print(f'Could not read file: {price_data_path}. Downloading data for "{ticker}" from yahoo.com...')
        price_data = web.DataReader(ticker, 'yahoo', start, end)
        price_data.to_csv(price_data_path)
    
    return price_data


def add_sma_column(data_frame, col_name, num_days) :
    """Calculate the Simple Moving Average (SMA) over num_days for col_name and add to data_frame."""  
    data_frame[f'{col_name} SMA{num_days}'] = data_frame[col_name].rolling(window=num_days).mean()


def get_pct_diff(a, b) :
    """Calcuate the percent difference between a and b. %diff = 100*(a-b)/b."""
    return 100*(a - b)/b


def add_sma_pct_diff_column(data_frame, col_name, sma_period) :
    """Calculate the percent difference between col_name and the Simple Moving Avg of col_name."""
    sma = data_frame[col_name].rolling(window=sma_period).mean()
    data_frame[f'pct diff {col_name} SMA{sma_period}'] = get_pct_diff(data_frame[col_name], sma)


def get_sma_pct_diff_df(data_frame, col_name, sma_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Return a dataframe containing col_name from data_frame and several columns of percent 
    difference between col_name and the Simple Moving Averages (SMAs) for various periods.
    """
    sma_pct_diff_df = pd.DataFrame(data_frame[col_name])
    for d in sma_periods_list :
        add_sma_pct_diff_column(sma_pct_diff_df, col_name, d)

    return sma_pct_diff_df


def get_sma_df(data_frame, col_name, sma_periods_list=[3, 5, 10, 20, 50, 100, 200]) :
    """Return a dataframe containing col_name from data_frame and several columns of Simple
    Moving Averages (SMAs) for various periods
    """
    sma_df = pd.DataFrame(data_frame[col_name])
    for d in sma_periods_list :
        add_sma_column(sma_df, col_name, d)

    return sma_df


def get_google_trends_sma_pct_diff_df(data_frame, search, sma_periods_list=[3,5,10,20,50,100,200]) :
    """Get a dataframe containing the percent difference of google search trends with respect
    to Simple Moving Averages of trend data of various periods."""
    trend = get_google_trends_df(data_frame, search)
    trend = get_sma_pct_diff_df(trend, search, sma_periods_list)
    del trend[search]
    return trend

def add_days_since_valid_date(data_frame, new_col_name) :
    """Return a new DataFrame which is a copy of data_frame with and added column called 
    new_col_name which contains the number of days elapsed since the last valid data. 
    The index of data_frame must be a date."""
    assert(isinstance(data_frame, pd.DataFrame))
    assert(isinstance(data_frame.index, pd.DatetimeIndex))

    elapsed_df = pd.DataFrame(data_frame)
    day_length = np.timedelta64(1, 'D')
    last_date = data_frame.index.values[0]
    days_elapsed = []

    for d in data_frame.index.values :
        days_since_last = (abs(d-last_date).astype('timedelta64[D]') / day_length).astype(int)
        days_elapsed.append(days_since_last)
        last_date = d

    elapsed_df[new_col_name] = days_elapsed
    return elapsed_df


def add_days_since_valid_value(data_frame, col_name, new_name_prefix):
    """Return a new DataFrame which is a copy of data_frame with an added column which gives   
    the days elapsed since col_name last had a valid value (i.e. not NaN).
    Newly added column will have the name 'f{new_name_prefix} {col_name}'.
    The index of data_frame must be a date."""
    assert(isinstance(data_frame, pd.DataFrame))
    assert(isinstance(data_frame.index, pd.DatetimeIndex))

    elapsed_df = pd.DataFrame(data_frame)
    day_length = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    days_elapsed = []

    for v,d in zip(data_frame[col_name].values, data_frame.index.values):
        if not np.isnan(v) :
            last_date = d
        days_since_nan = ((d-last_date).astype('timedelta64[D]') / day_length).astype(int)
        days_elapsed.append(days_since_nan)
        
    elapsed_df[f'{new_name_prefix} {col_name}'] = days_elapsed
    return elapsed_df


def get_google_trends_df(data_frame, search):
    """Get a dataframe with google trends for a search.""" 
    # create data_range
    date_range = [f'{get_df_start_date(data_frame)} {get_df_end_date(data_frame)}']
    
    # Set up the trend fetching object
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [search]

    try:
        # Create the search object
        pytrends.build_payload(kw_list, cat=0, timeframe=date_range[0], geo='', gprop='news')
        
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
    trends = add_days_since_valid_value(trends, search, 'Days since updated')
    # clean up na values from upsample
    trends = pd.DataFrame.fillna(trends, method='ffill')
    
    return trends