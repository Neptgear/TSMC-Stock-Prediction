import sys
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_fetch import _get_alpha_vantage_key

key = _get_alpha_vantage_key()
print('key present:', bool(key))
url = 'https://www.alphavantage.co/query'
params = {
    'function':'TIME_SERIES_DAILY_ADJUSTED',
    'symbol':'TSM',
    'outputsize':'full',
    'apikey': key,
}
r = requests.get(url, params=params, timeout=30)
print('status:', r.status_code)
print('len text:', len(r.text))
print('snippet:', r.text[:200].replace('\n',' '))
