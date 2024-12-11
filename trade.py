import time
import logging
from datetime import datetime
import os
from binance.client import Client as bn
import ccxt

# track runtime
start_time = time.time()

# streamline output
log_file_path = f"/home/dev/code/tmp/{datetime.now():%Y-%m-%d %H-%M-%S}.txt"
logging.basicConfig(filename = log_file_path, level = logging.INFO, format = '%(message)s')

# binance
try:
    bn_api = bn(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"), tld = "us")
    bn_nfo = bn_api.get_exchange_info()
    bn_assets = {symbol['baseAsset'] for symbol in bn_nfo['symbols']}
    logging.info(f"{bn_assets}\n{len(bn_assets)} Binance\n")
except Exception as e:
    bn_assets = set()
    logging.info(f"Binance Assets {e}\n")   

# coinbase
try:
    cb_api = ccxt.coinbase({'apiKey': os.getenv("COINBASE_API_KEY"), 'secret': os.getenv("COINBASE_API_SECRET")})
    cb_assets = set(cb_api.fetch_currencies().keys())
    logging.info(f"{cb_assets}\n{len(cb_assets)} Coinbase\n")
except Exception as e:
    cb_assets = set()
    logging.info(f"Coinbase Assets {e}\n")

# common
if bn_assets and cb_assets:
    shared = bn_assets & cb_assets
    logging.info(f"{shared}\n{len(shared)} Shared\n")

# calculate runtime
end_time = time.time()
seconds = end_time - start_time
logging.info(f"\nRuntime")
logging.info(f"{round(seconds, 2)} seconds")
logging.info(f"{round((seconds / 60), 2)} minutes")

