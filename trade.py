import time
import logging
from datetime import datetime
import os
import requests
from binance.client import Client as bn
import ccxt
import asyncio
from aiohttp import ClientSession
import pandas as pd
import numpy as np
import polars as pl
import talib
pd.set_option('display.max_columns', None)


# Shared Binance and Coinbase assets
def get_cryptos():
    # Binance
    try:
        bn_api = bn(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"), tld = "us")
        bn_nfo = bn_api.get_exchange_info()
        bn_assets = {symbol['baseAsset'] for symbol in bn_nfo['symbols']}
        logging.info(f"{bn_assets}\n{len(bn_assets)} Binance\n")
    except Exception as e:
        bn_assets = set()
        logging.info(f"get_cryptos() Binance {e}\n")
        return 1

    # Coinbase
    try:
        cb_api = ccxt.coinbase({'apiKey': os.getenv("COINBASE_API_KEY"), 'secret': os.getenv("COINBASE_API_SECRET")})
        all_assets = cb_api.fetch_currencies() 

        # No asset_id means defi, delisted, mobile only, etc
        cb_assets = {asset for asset, details in all_assets.items() if 'asset_id' in details['info']}
        logging.info(f"{cb_assets}\n{len(cb_assets)} Coinbase\n")
    except Exception as e:
        cb_assets = set()
        logging.info(f"get_cryptos() Coinbase {e}\n")
        return 1
    
    # Filter stablecoins, delisted, etc
    remove = {"USDC", "USDT", "SPELL", "HNT", "JASMY", "GAL", "WBTC", "REP", "AMP"}
    logging.info(f"{remove}\n{len(remove)} Invalid\n")

    # Common
    if bn_assets and cb_assets:
        shared = (bn_assets & cb_assets) - remove
        logging.info(f"{shared}\n{len(shared)} Shared\n")
        return shared
    else:
        return 1


# Spot price for one asset
async def get_price(asset, session):
    try:
        url = f"https://api.coinbase.com/v2/prices/{asset}-USD/spot"
        async with session.get(url) as response:
            data = await response.json()
            if 'data' in data and 'amount' in data['data']:
                return asset, float(data['data']['amount'])
            else:
                return asset, None
    except Exception as e:
        logging.info(f"fetch_price({asset}) {e}\n")
        return asset, None


# Spot prices for multiple assets
async def get_all_prices(assets):
    try:
        async with ClientSession() as session:
            tasks = {get_price(asset, session) for asset in assets}
            results = await asyncio.gather(*tasks)
            prices = {asset: price for asset, price in results if price is not None}
            logging.info(f"{prices}\n{len(prices)} Prices\n")
            return prices
    except Exception as e:
        logging.info(f"get_all_prices() {e}\n")


# Candlestick data for a single asset
async def fetch_candles(symbol, interval, session):
    url = f"https://api.binance.us/api/v3/klines"
    params = {
        "symbol": symbol + "USDT",
        "interval": interval,
    }
    try:
        async with session.get(url, params = params) as response:
            data = await response.json()
            if response.status == 200:
                # Convert to dataframe and remove ignore column
                columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                           'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume', 
                           'Taker Buy Quote Volume', 'Ignore']
                df = pd.DataFrame(data, columns = columns)
                df = df.drop('Ignore', axis = 1)
                
                # Convert numeric features to float
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                    'Taker Buy Base Volume', 'Taker Buy Quote Volume']
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

                # Convert timestamps to readable datetime
                df['Open Time'] = pd.to_datetime(df['Open Time'], unit = 'ms')
                df['Close Time'] = pd.to_datetime(df['Close Time'], unit = 'ms')
                
                # Normalized y value over raw price
                df['y'] = np.log(1 + df['Close'].pct_change()) 
                # logging.info(f"{symbol}\n{df}\n")     # Output after technical analysis
                
                # Check for new assets
                # If fewer than five hundred rows than there's not enough historical data
                if len(df) < 500:
                    logging.info(f"{symbol} New Asset Removed\n{df}\n")
                    return symbol, None
                else:
                    return symbol, df
            else:
                logging.info(f"fetch_candles({symbol}) No Candles\n")
                return symbol, None
    except Exception as e:
        logging.info(f"fetch_candles({symbol} {interval}) {e}\n")
        return symbol, None


# Candlestick data for multiple assets
async def fetch_all_candles(symbols, interval):
    try:
        async with ClientSession() as session:
            tasks = [fetch_candles(symbol, interval, session) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            
            # Create a dictionary of DataFrames for each symbol
            candles = {symbol: df for symbol, df in results if df is not None}
            logging.info(f"{len(candles)} DataFrames\n")
            return candles
    except Exception as e:
        logging.info(f"fetch_all_candles() {e}\n")
        return 1


# Custom ema function for technical analysis indicator calculations
def calculate_ema(polars_series, span):
    try:
        alpha = 2 / (span + 1)
        
        # Initialize the ema with the first value
        ema_values = [polars_series[0]]
        for price in polars_series[1:]:
            # Compute the next ema value based on the previous ema value
            next_ema = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(next_ema)
        return pl.Series(ema_values)
    except Exception as e:
        return None


# Calculate technical analysis indicators from peer reviewed article
# [ma crosses, rsi, kdj, macd top and bottom structures, macd golden and dead crosses]
# Trade signals -> 0 is neutral, 1 is buy and 2 is sell
def calc_ta(asset, df):
    try:
        # Convert to polars for parallel processing and multithreading
        polarsDF = pl.DataFrame(df)

        # 1/5 moving average crosses
        close_prices = polarsDF['Close'].to_numpy()                         # convert to numpy arrays for compatibility and performance
        short_ma = pl.Series(talib.SMA(close_prices, timeperiod = 5))       # 5 period moving average
        long_ma = pl.Series(talib.SMA(close_prices, timeperiod = 10))       # 10 period moving average
        polarsDF = polarsDF.with_columns([pl.Series('short_ma', short_ma), pl.Series('long_ma', long_ma)]) 
        
        ma_signal = np.where((polarsDF['short_ma'] > polarsDF['long_ma']) & (polarsDF['short_ma'].shift(1) <= polarsDF['long_ma'].shift(1)), 1, np.where((polarsDF['short_ma'] < polarsDF['long_ma']) & (polarsDF['short_ma'].shift(1) >= polarsDF['long_ma'].shift(1)), 2, 0))
        polarsDF = polarsDF.with_columns(pl.Series(name = 'ma_signal', values = ma_signal))

        # 2/5 rsi
        short_rsi = pl.Series(talib.RSI(close_prices, timeperiod = 5))
        long_rsi = pl.Series(talib.RSI(close_prices, timeperiod = 10))
        
        # Signal generation based on rsi levels
        buy_signal = np.where((short_rsi < 50) & (short_rsi.shift(1) > long_rsi.shift(1)), 1, 0)
        sell_signal = np.where((short_rsi > 50) & (short_rsi.shift(1) < long_rsi.shift(1)), 2, 0)
        
        # Combine buy and sell signals to create final rsi signals and add to dataframe
        rsi_signals = np.where(buy_signal == 1, 1, np.where(sell_signal == 2, 2, 0))
        polarsDF = polarsDF.with_columns([pl.Series('short_rsi', short_rsi), pl.Series('long_rsi', long_rsi),pl.Series('rsi_signal', rsi_signals)])

        # 3/5 kdj
        # Calculate row stochastic value
        high_prices = polarsDF['High'].to_numpy()
        low_prices = polarsDF['Low'].to_numpy()
        close9 = talib.EMA(close_prices, timeperiod = 9)
        high9 = talib.MAX(high_prices, timeperiod = 9)      # Highest price within nine periods
        low9 = talib.MIN(low_prices, timeperiod = 9)        # Lowest price within nine periods
       
        # Handle divide by zero issues
        denominator = high9 - low9
        rsv = np.empty_like(close9)
        rsv.fill(50)    # Pre-fill with neutral stochastic value fifty
        
        # Only perform the division for entries where denominator isn't zero
        mask = denominator != 0
        rsv[mask] = ((close9[mask] - low9[mask]) / denominator[mask]) * 100

        # Calculate k, d, j
        k = talib.EMA(rsv, timeperiod=3)
        d = talib.EMA(k, timeperiod=3)
        j = 3 * k - 2 * d
        
        # Buy and sell signals
        buy_kdj = (k < 25) & (d < 25)
        sell_kdj = (k > 75) & (d > 75)

        # Combine buy and sell signals
        kdj_signals = np.zeros_like(k, dtype=int)   # Initialize with zeros
        kdj_signals[buy_kdj] = 1                    # Set buy signals
        kdj_signals[sell_kdj] = 2                   # Set sell signals

        # Add kdj signals to the polars dataframe
        polarsDF = polarsDF.with_columns([pl.Series('k', k), pl.Series('d', d), pl.Series('j', j), pl.Series('kdj_signal', kdj_signals)])

        # 4/5 macd top and bottom structures
        # Calculate ema12, ema26, dif, dea, and macd
        ema12 = calculate_ema(close_prices, 12)
        ema26 = calculate_ema(close_prices, 26)
        dif = ema12 - ema26
        dea = calculate_ema(dif, 9)
        macd = (dif - dea) * 2

        # Add the calculated columns
        polarsDF = polarsDF.with_columns([pl.Series('ema12', ema12), pl.Series('ema26', ema26), pl.Series('dif', dif), pl.Series('dea', dea), pl.Series('macd', macd)])
        
        # Identify rising and falling waves
        rising_wave = (macd > 0) & (macd.shift(1) > 0)
        rising_wave_int = rising_wave.map_elements(lambda x: 1 if x else 0, return_dtype = pl.Int32)
        falling_wave = (macd < 0) & (macd.shift(1) < 0)
        falling_wave_int = falling_wave.map_elements(lambda x: 1 if x else 0, return_dtype = pl.Int32)

        # Identify top and bottom structures
        # Must convert to int after pandas
        close = polarsDF['Close']
        top_condition = ((close > close.shift())        # Current close price is higher than previous close price
            & (macd > macd.shift())                     # Current macd is higher than previous macd
            & (macd < macd.shift(2)))                   # Current macd is lower than macd two intervals ago

        bottom_condition = ((close < close.shift())     # Current close price is lower than previous close price
            & (macd < macd.shift())                     # Current macd is lower than previous macd
            & (macd > macd.shift(2)))                   # Current macd is higher than macd two intervals ago

        polarsDF = polarsDF.with_columns([pl.Series('rising_wave', rising_wave_int), pl.Series('falling_wave', falling_wave_int), pl.Series('top_structures', top_condition), pl.Series('bottom_structures', bottom_condition)]) 
        
        # Golden and dead cross signals
        golden_cross = [0] * len(polarsDF)  # Pre-fill with 0s
        dead_cross = [0] * len(polarsDF)    # Pre-fill with 0s

        # Convert dif and dea to numpy arrays for iteration
        dif_array = dif.to_numpy()
        dea_array = dea.to_numpy()

        # Iterate over the data starting from the second element
        for i in range(1, len(dif_array)):
            # Check for golden cross
            if (dif_array[i] > dea_array[i] and dif_array[i-1] < dea_array[i-1] and dif_array[i] > 0 and dea_array[i] > 0):
                golden_cross[i] = 1

            # Check for dead cross
            if (dif_array[i] < dea_array[i] and dif_array[i-1] > dea_array[i-1] and dif_array[i] < 0 and dea_array[i] < 0):
                dead_cross[i] = 2

        # Convert lists to polars Series
        golden_cross_series = pl.Series(golden_cross)
        dead_cross_series = pl.Series(dead_cross)   

        # Add columns to polarsDF
        polarsDF = polarsDF.with_columns([pl.Series('golden', golden_cross_series), pl.Series('dead', dead_cross_series)
])

        # Polars to pandas
        polars_df = polarsDF.to_pandas().dropna()
        polars_df.dropna(inplace = True)

        # Convert boolean columns to integers to compensate for polars to pandas data conversion issues
        boolean_columns = ['top_structures', 'bottom_structures', 'rising_wave', 'falling_wave', 'golden', 'dead']
        for col in boolean_columns:
            polars_df[col] = polars_df[col].astype(int)          
        
        logging.info(f"{asset}\n{polars_df}\n")
        return polars_df

    except Exception as e:
        return 1


##########
## MAIN ##
##########

# Track runtime
start_time = time.time()

# Streamline output
log_file_path = f"/home/dev/code/tmp/{datetime.now():%Y-%m-%d %H-%M-%S}.txt"
logging.basicConfig(filename = log_file_path, level = logging.INFO, format = '%(message)s')

# Assets, candles, technical analysis and prices
# Only proceed to next function if previous doesn't error out
names = get_cryptos()
if names != 1:
    candles = asyncio.run(fetch_all_candles(names, '1h'))
if candles != 1:
    ta_candles = {symbol: calc_ta(symbol, df) for symbol, df in candles.items()}
if ta_candles != 1:
    prices = asyncio.run(get_all_prices(candles.keys()))

# Calculate runtime
end_time = time.time()
seconds = end_time - start_time
logging.info(f"\nRuntime")
logging.info(f"{round(seconds, 2)} seconds")
logging.info(f"{round((seconds / 60), 2)} minutes")

