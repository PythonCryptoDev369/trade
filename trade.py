import time
import logging
from datetime import datetime
import os
import requests
from binance.client import Client as bn
import ccxt
import asyncio
from aiohttp import ClientSession


def get_cryptos():
    # Binance
    try:
        bn_api = bn(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"), tld = "us")
        bn_nfo = bn_api.get_exchange_info()
        bn_assets = {symbol['baseAsset'] for symbol in bn_nfo['symbols']}
        logging.info(f"{bn_assets}\n{len(bn_assets)} Binance\n")
    except Exception as e:
        bn_assets = set()
        logging.info(f"Binance Assets {e}\n")   

    # Coinbase
    try:
        cb_api = ccxt.coinbase({'apiKey': os.getenv("COINBASE_API_KEY"), 'secret': os.getenv("COINBASE_API_SECRET")})
        all_assets = cb_api.fetch_currencies() 
        
        # No asset_id means defi, delisted, mobile only, etc
        cb_assets = {asset for asset, details in all_assets.items() if 'asset_id' in details['info']}
        logging.info(f"{cb_assets}\n{len(cb_assets)} Coinbase\n")
    except Exception as e:
        cb_assets = set()
        logging.info(f"Coinbase Assets {e}\n")

    # Common
    if bn_assets and cb_assets:
        shared = bn_assets & cb_assets
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
        logging.error(f"fetch_price({asset}) {e}")
        return asset, None


# Spot prices for multiple assets
async def get_all_prices(assets):
    async with ClientSession() as session:
        tasks = [get_price(asset, session) for asset in assets]
        results = await asyncio.gather(*tasks)
        prices = {asset: price for asset, price in results if price is not None}
        logging.info(f"{prices}\n{len(prices)} Prices\n")
        return prices


##########
## MAIN ##
##########

# Track runtime
start_time = time.time()

# Streamline output
log_file_path = f"/home/dev/code/tmp/{datetime.now():%Y-%m-%d %H-%M-%S}.txt"
logging.basicConfig(filename = log_file_path, level = logging.INFO, format = '%(message)s')

# Available assets and prices
names = get_cryptos()
prices = asyncio.run(get_all_prices(names))

# Calculate runtime
end_time = time.time()
seconds = end_time - start_time
logging.info(f"\nRuntime")
logging.info(f"{round(seconds, 2)} seconds")
logging.info(f"{round((seconds / 60), 2)} minutes")

