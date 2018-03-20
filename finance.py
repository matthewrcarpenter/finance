"""Financial utility functions."""
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

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

    days_since_col = 'Days Since Trading'
    data = create_days_since_valid_date(data_frame, days_since_col)
    # First row will have 0, replace with 1
    data.loc[data[days_since_col] == 0, days_since_col] = 1

    # Get the days until next day of trading. Reverse sort data, call same 
    # function as above, then resort in normal order.
    data.sort_index(ascending=False, inplace=True)
    days_until_col = 'Days Until Trading'
    data = create_days_since_valid_date(data, days_until_col)
    data.sort_index(ascending=True, inplace=True)
    # Last row will have 0, replace with 1
    data.loc[data[days_until_col] == 0, days_until_col] = 1

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


def get_price_data(ticker, update=True) :
    """Get a DataFrame with the price data for ticker. Rows with NA values will 
    be removed from data."""
    # Defualt to yahoo for now
    data = get_price_data_yahoo(ticker, update).dropna()
    
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


def get_price_data_yahoo(ticker, update) :
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
        if (update) :
            price_data = update_price_data_yahoo(price_data, ticker)
    except :
        print(f'Could not read file: {price_data_path}. ' 
            f'Downloading data for "{ticker}" from yahoo.com...')
        price_data = web.DataReader(ticker, 'yahoo', start, end)
        price_data.to_csv(price_data_path)
    
    return price_data


def use_data_frame_if_inplace(data_frame, inplace) :
    """Convenience methode that returns data_frame if inplace==True, else 
    returns a copy of data_frame."""
    if inplace :
        return data_frame
    else :
        return data_frame.copy()



def add_bollinger_bands(data_frame, column, num_days=20, inplace=False) :
    """Adds Bollinger Bands columns to a DataFrame.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
 
    df = use_data_frame_if_inplace(data_frame, inplace)

    # Calculate SMA, then get SMA as Series
    sma = add_sma_column(df, column, num_days, inplace=True)
    sma = sma[f'{column} SMA{num_days}']
    # ... calculate rolling Standard Deviation of values
    msd = pd.Series(df[column]).rolling(window=num_days, center=False).std()

    # Create Hi and Lo Bollinger Bands
    df[f'{column} BBandHi{num_days}'] = sma + (2 * msd)
    df[f'{column} BBandLo{num_days}'] = sma - (2 * msd)

    return df


def add_ema_column(data_frame, column, num_days, inplace=False) :
    """Add a column with Exponential Moving Average (EMA) data to a DataFrame.

    Args:
        data_frame: DataFrame containing data of interest.
        column: String containing column of interest. Column is assumed to 
            contain daily data.
        num_days: Number of days (span) to use for calculating EMA.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
 
    df = use_data_frame_if_inplace(data_frame, inplace)

    df[f'{column} EMA{num_days}'] = df[column].ewm(span=num_days).mean()
    return df


def add_ema_columns(data_frame, column, ema_periods_list=[12, 26], 
    inplace=False) :
    """Add several columns to a DataFrame containing the Exponential Moving 
    Averages (EMAs) of those values for various periods.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)
    for d in ema_periods_list :
        add_ema_column(df, column, d, inplace=True)
    return df


def add_ema_pct_diff_column(data_frame, column, ema_period, inplace=False) :
    """Add a column with the percent difference between base values and the 
    Exponential Moving Average of those values."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)

    ema = df[column].ewm(span=ema_period).mean()
    df[f'pct diff {column} EMA{ema_period}'] = get_pct_diff(df[column], ema)

    return df


def add_ema_pct_diff_columns(data_frame, column, ema_period, 
        ema_periods_list=[12,26], inplace=False) :
    """Add columns with the percent difference between base values and the 
    Exponential Moving Average of those values."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)
    for d in ema_periods_list :
        add_ema_pct_diff_column(df, column, d, inplace=True)

    return df


def add_macd_columns(data_frame, column, inplace=False) :
    """Add MACD (Moving Average Convergence/Divergence) columns to a DataFrame.
    
    Args:
        data_frame: DataFrame of interest.
        column: String containing the column name with the data of interest.
        inplace: Boolean indicating whether to modify data_frame inplace.

    Returns:
        DataFrame with added columns for: 'EMA12', 'EMA26', 'MACD', 
        'MACD EMA9', 'MACD Hist'. All of the listed columns will be prepended 
        with the column argument (e.g., f'{column} EMA12'). 
    """
    fast = 12
    slow = 26

    FAST = f'{column} EMA{fast}'
    SLOW = f'{column} EMA{slow}'
    MACD = f'{column} MACD'
    SIGNAL = f'{column} MACD EMA9'
    HIST = f'{column} MACD Hist'

    df = use_data_frame_if_inplace(data_frame, inplace)
    
    df[FAST] = df[column].ewm(span=12).mean()
    df[SLOW] = df[column].ewm(span=26).mean()
    df[MACD] = df[FAST] - df[SLOW]
    df[SIGNAL] = df[MACD].ewm(span=9).mean()
    df[HIST] = df[MACD] - df[SIGNAL]

    return df    


def add_sma_column(data_frame, column, num_days, inplace=False) :
    """Add a column with Simple Moving Average (SMA) data to a DataFrame."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)

    df[f'{column} SMA{num_days}'] = df[column].rolling(
        window=num_days).mean()
    
    return df


def add_sma_columns(data_frame, column, sma_periods_list=[12, 26], 
    inplace=False) :
    """Add several columns to a DataFrame containing the Simple Moving Averages 
    (SMAs) of values in column for various periods."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)
    for d in sma_periods_list :
        add_sma_column(df, column, d, inplace=True)
    return df


def add_sma_pct_diff_column(data_frame, column, sma_period, inplace=False) :
    """Add a column with the percent difference between data and the Simple 
    Moving Average of that data."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)

    sma = data_frame[column].rolling(window=sma_period).mean()
    df[f'pct diff {column} SMA{sma_period}'] = get_pct_diff(df[column], sma)

    return df


def add_sma_pct_diff_columns(data_frame, column, sma_periods_list=[6, 12],
        inplace=False) :
    """Add several columns to a DataFrame containing the percent difference 
    between those values and the Simple Moving Averages (SMAs) of those values 
    for various periods."""
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"

    df = use_data_frame_if_inplace(data_frame, inplace)
    for d in sma_periods_list :
        add_sma_pct_diff_column(df, column, d, inplace=True)
    return df


def get_pct_diff(measured, expected) :
    """Calcuate the percent difference between measured and expected values."""
    return 100*(measured - expected)/expected


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


def create_days_since_valid_date(data_frame, new_column) :
    """Copy a DataFrame and add a column containing the number of days elapsed
    since the last valid date. 
    
    Args:
        data_frame: DataFrame with a DatetimeIndex.
        new_column: String with the name of the new column containg the days
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

    elapsed_df[new_column] = days_elapsed
    return elapsed_df


def create_days_since_valid_value(data_frame, column, new_name_prefix):
    """Copy a DataFrame and add a column which gives the days elapsed since the
    column of interest had a valid value.

    Args:
        data_frame: DataFrame of interest, must have DatetimeIndex.
        column: String with name of column of interest in data_frame.
        new_name_prefix: String with a prefix that will be added to column to 
            create new column. Newly added column will have the name 
            'f{new_name_prefix} {column}'.
    """
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    assert isinstance(data_frame.index, pd.DatetimeIndex), \
        "data_frame must use a DatetimeIndex"

    elapsed_df = pd.DataFrame(data_frame)
    day_length = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    days_elapsed = []

    for v,d in zip(data_frame[column].values, data_frame.index.values):
        if not pd.isna(v) :
            last_date = d
        days_since_valid = ((d-last_date).astype('timedelta64[D]') /
            day_length).astype(int)
        days_elapsed.append(days_since_valid)
        
    elapsed_df[f'{new_name_prefix} {column}'] = days_elapsed
    return elapsed_df


def get_google_trends_df(data_frame, search):
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


def create_fitted_line_df(data_frame, column, fitted_column) :
    '''Create a DataFrame with base values of interest plus a column with a best
    fit line for those values.

    Args:
        data_frame: DataFrame of interest, must use a DatetimeIndex.
        column: String with name of the column of interest. Any NA values in
            column will be filled in with predicted values in fitted_column.
        fitted_column: String with the name of the new column that contains 
            the best fit line values. 
    Returns:
        A DataFrame containing a column with the values of interest plus a 
        column with the best fit line values.
    '''
    assert isinstance(data_frame, pd.DataFrame), \
        "data_frame must be pandas.DataFrame object"
    assert isinstance(data_frame.index, pd.DatetimeIndex), \
        "data_frame must use a DatetimeIndex"
    assert (column is not None)
    assert (fitted_column is not None)
    
    df = pd.DataFrame(data_frame[column])
    
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
    y = fit_df[column]
    
    # Fit the values, and get the prediction function
    model = np.polyfit(x, y, deg=1)
    predict = np.poly1d(model)
    
    # get predicted values for original data_frame
    fit_vals = predict(df['index'])
    df[fitted_column] = fit_vals
    
    #return_df = pd.DataFrame(data_frame[column])
    #return_df = return_df.join(df, how='left')
    
    # delete the index values and the original data
    del df['index']
    del df[column]
    
    # now we'll join back to the original data_frame to eliminate the extra 
    # data added by the resampling
    df2 = pd.DataFrame(data_frame[column])
    df = df2.join(df, how='inner')
    del df[column]
        
    return df


def plot_daily_ticker(ohlcv, title=None, macd=None, rsi=None, overlay=None) :
    """Plots a daily candlestick chart for ticker OHLCV data.

    This code is based on 
    https://matplotlib.org/1.5.1/examples/pylab_examples/finance_work2.html
    
    Args:
        ohlcv: DataFrame containing OHLCV data. Must contain date index and 
            columns name 'Open', 'High', 'Low', 'Close', and 'Volume'.
        macd: DataFrame containing the MACD columns associated with ohlcv.
        rsi: DataFrame containing the RSI column associated with ohlcv
        overlay: DataFrame containing other columns to overlay onto the 
        OHLC plot. All columns in overlay will be plotted. 
    """
    assert isinstance(ohlcv, pd.DataFrame), \
        "ohlcv must be pandas.DataFrame object"
    
    # Get start and end dates for date axis, pad for plotting
    start_date = ohlcv.index[0] - dt.timedelta(0.5)
    end_date = ohlcv.index[-1] + dt.timedelta(0.5)
    label_axis = None
    no_label_axes = []
    
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

    fig = plt.figure(facecolor='white')
    axescolor = '#f6f6f6'  # the axes background color
    textsize = 9
    left, width = 0.1, 0.8
  
    rect_rsi = [left, 0.7, width, 0.1]
    rect_ohlcv = [left, 0.1, width, 0.6]
    rect_macd = [left, 0.0, width, 0.1]

    ax_rsi = None
    
    if title is None :
        title=''

    # RSI
    if rsi is not None :
        ax_rsi = fig.add_axes(rect_rsi, 
            facecolor=axescolor)  # left, bottom, width, height
        format_rsi_axis(ax_rsi, rsi[start_date:end_date])
        no_label_axes.append(ax_rsi)
        ax_rsi.set_title(f'{title}')

    # OHLC
    ax_ohlc = fig.add_axes(rect_ohlcv, facecolor=axescolor, sharex=ax_rsi)
    if overlay is not None:
        overlays = overlay[start_date:end_date]
    else :
        overlays = None
    format_ohlc_axis(ax_ohlc, ohlcv, overlays)
    if rsi is None :
        ax_ohlc.set_title(f'{title}')

    # Volume
    ax_vol = ax_ohlc.twinx()
    format_volume_axis(ax_vol, ohlcv)
    no_label_axes.append(ax_vol)

    # ... summary string 
    s = get_plot_ohlcv_summary_str(ohlcv)
    ax_ohlc.text(0.3, 0.95, s, transform=ax_ohlc.transAxes, 
        fontsize=textsize+1)

    # ... legend
    props = font_manager.FontProperties(size=10)
    leg = ax_ohlc.legend(loc='best', prop=props) #, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    # MACD
    ax_macd = None
    if macd is not None :
        ax_macd = fig.add_axes(rect_macd, facecolor=axescolor, sharex=ax_ohlc)
        format_macd_axis(ax_macd, macd[start_date:end_date])
        no_label_axes.append(ax_ohlc)
        label_axis = ax_macd
    else :
        label_axis = ax_ohlc
    
    # turn off upper axis tick labels
    for ax in no_label_axes :
        for label in ax.get_xticklabels():
            label.set_visible(False)

    # rotate the lower ones, etc
    for label in label_axis.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    label_axis.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    # set start and end date for all axes
    all_axes = list(no_label_axes)
    all_axes.append(label_axis)
    for ax in all_axes :
        ax.set_xlim(start_date, end_date)
    
    class MyLocator(mticker.MaxNLocator):
        def __init__(self, *args, **kwargs):
            mticker.MaxNLocator.__init__(self, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            return mticker.MaxNLocator.__call__(self, *args, **kwargs)

    # at most 5 ticks, pruning the upper and lower so they don't overlap
    # with other ticks
    ax_ohlc.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
    if (ax_macd) :
        ax_macd.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))

    ax_ohlc.yaxis.set_major_locator(MyLocator(5, prune='both'))
    if (ax_macd) :
        ax_macd.yaxis.set_major_locator(MyLocator(5, prune='both'))

    plt.show()


def format_rsi_axis(axis, df_rsi) :
    """Convenience fn used by plot_daily_ticker(): Format the RSI axis of 
    ticker plot. 
    """
    textsize = 9
    fillcolor = 'darkgoldenrod'
    date = df_rsi.index
    rsi = df_rsi['RSI']

    axis.plot(date, rsi, color=fillcolor)
    axis.axhline(70, color='red', alpha=0.3)
    axis.axhline(30, color='green', alpha=0.3)
    axis.fill_between(date, rsi, 70, where=(rsi >= 70), facecolor='red', 
        edgecolor='red', alpha=0.3)
    axis.fill_between(date, rsi, 30, where=(rsi <= 30), facecolor='green', 
        edgecolor='green', alpha=0.3)
    axis.text(0.6, 0.9, '>70 = overbought', va='top', transform=axis.transAxes, 
        fontsize=textsize)
    axis.text(0.6, 0.1, '<30 = oversold', transform=axis.transAxes, 
        fontsize=textsize)
    axis.set_ylim(0, 100)
    axis.set_yticks([30, 70])
    axis.text(0.025, 0.95, 'RSI (14)', va='top', transform=axis.transAxes, 
        fontsize=textsize)


def format_ohlc_axis(axis, ohlc_df, overlays_df) :
    """Convenience fn used by plot_daily_ticker(): Format the OHLC axis for a 
    plot.
    """
    # convenience vars
    date = ohlc_df.index
    low = ohlc_df['Low']
    high = ohlc_df['High']
    op = ohlc_df['Open']
    cl = ohlc_df['Close']
    
    # get days that were up
    up = cl >= op

    # Adjust width of candlestick appropriately
    # FIXME: this is currently a hack based on typical plot size
    if (len(ohlc_df) < 100) :
        width = 5
        alpha = 1.0
    elif (len(ohlc_df) < 150) :
        width = 3
        alpha = 1.0
    else :
        width = 1
        alpha = 0.4
        
    # high and low
    axis.vlines(date[up], low[up], high[up], color='green', label='_nolegend_',
        alpha=alpha)
    axis.vlines(date[~up], low[~up], high[~up], color='red', label='_nolegend_',
        alpha=alpha)
    # open and close
    axis.vlines(date[up], op[up], cl[up], color='green', label='_nolegend_', 
        linewidths=width)
    axis.vlines(date[~up], op[~up], cl[~up], color='red', label='_nolegend_', 
        linewidths=width)

    axis.plot(date, cl, color='black', label='Close', alpha=0.3)
    
    # Plot overlays
    if overlays_df is not None:
        overlays_df.plot(ax=axis)


def format_volume_axis(axis, ohlcv_df) :
    """Convenience fn used by plot_daily_ticker(): Format the volume axis for 
    a plot.
    """
    r = ohlcv_df
    fillcolor = 'darkgoldenrod'
    volume = (r['Close']*r['Volume'])/1e6  # dollar volume in millions
    vmax = volume.max()
    date = ohlcv_df.index
    
    axis.fill_between(date, volume, 0, label='Volume', facecolor=fillcolor, 
        edgecolor=fillcolor, alpha=0.1)
    axis.set_ylim(0, 1.05*vmax)
    axis.set_yticks([])


def get_plot_ohlcv_summary_str(ohlcv_df) :
    """Gets a one-line summary string for a range of OHLCV data.
    """
    first = ohlcv_df.iloc[0]
    last = ohlcv_df.iloc[-1]
    
    start_date = ohlcv_df.index[0]
    end_date = ohlcv_df.index[-1]
    date_format_str = '%d-%b-%Y'
    op = first['Open']
    cl = last['Close']
    high = ohlcv_df['High'].max()
    low = ohlcv_df['Low'].min()
    volume = ohlcv_df['Volume'].sum()
    
    s = '%s to %s: O:%1.2f H:%1.2f L:%1.2f C:%1.2f V:%1.1fM Chg:%+1.2f%%' 
    s = s % (start_date.strftime(date_format_str), 
        end_date.strftime(date_format_str),
        op, high, low, cl, volume*1e-6, cl - op)
    return s


def format_macd_axis(axis, df_macd) :
    """Convenience fn used by plot_daily_ticker(): Format the MACD axis of a 
    plot.
    """
    # compute the MACD indicator
    fillcolor = 'darkslategrey'
    textsize = 9
    
    date = df_macd.index
    macd = df_macd['MACD']
    signal = df_macd['Signal']
    hist = df_macd['Histogram']
    
    axis.plot(date, macd, color='black', lw=1, alpha=0.3)
    axis.plot(date, signal, color='blue', lw=1, alpha=0.3)
    axis.fill_between(date, hist, 0, alpha=0.5, facecolor=fillcolor, 
        edgecolor=fillcolor)

    axis.text(0.025, 0.95, 'MACD', va='top', transform=axis.transAxes, 
        fontsize=textsize)
