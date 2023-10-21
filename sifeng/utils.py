import duckdb
import requests
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from pandas.tseries.offsets import MonthEnd

from typing import Union, Literal, List

__all__ = [
    "basic_info",
    "kline_day",
]

baseurl = "https://data.sifeng.site/"

def basic_info(fields: Union[Literal["*"], List[str]] = "*",
               stock_code: Union[Literal["*"], List[str]] = "*",
               local_dir: Path = Path.home() / ".sifeng/parquet/status_charts/",
               force_update: bool = False):
    """Fetch basic info from server

    Parameters
    ----------
    fields: Union[Literal["*"], List[str]], default `'*'`
        The fields you want, you may choose from `['stock_code', 'stock_name', 'area', 'industry',\
    'sector', 'list_status', 'list_date', 'st_flag']`, or enter `'*'` to choose all.
    stock_code: Union[Literal["*"], List[str]], default `'*'`
        The code of the stocks you want to query, or enter `'*'` for all.
    local_dir: pathlib.Path, default `Path.home() / ".sifeng/parquet/status_charts/"`
        The local path you want to data to download to, please do not change under normal circumsta
    nces.
    force_update: bool, default `False`
        Whether to force fetching from remote server. If `False`, then will only fetch if hadn't
    fetched before.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    filename = "BASIC-INFO.parquet"
    update = not (local_dir / filename).exists()
    if force_update:
        update = True
    if update:
        resp = requests.get(baseurl + filename)
        with open(local_dir / filename, "wb") as file:
            file.write(resp.content)
    if fields == "*":
        fields = ['stock_code', 'stock_name', 'area', 'industry', 'sector', 'list_status', 'list_date', 'st_flag']
    if stock_code == "*":
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / 'BASIC-INFO.parquet'}')"
    elif isinstance(stock_code, str):
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / 'BASIC-INFO.parquet'}') WHERE stock_code = '{stock_code}'"
    else:
        connector = "', '"
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / 'BASIC-INFO.parquet'}') WHERE stock_code IN ('{connector.join(stock_code)}')"
    return duckdb.sql(sql).df()

def kline_day(fields: Union[Literal["*"], List[str]] = "*",
              begin_date: str = "1999-12-19",
              end_date: str = datetime.now().strftime("%Y-%d-%m"),
              stock_code: Union[Literal["*"], List[str]] = "*",
              local_dir: Path = Path.home() / ".sifeng/parquet/kline_day/",
              n_jobs: int = 1):
    """Fetch kline (freq. day) from server

    Parameters
    ----------
    fields: Union[Literal["*"], List[str]], default `'*'`
        The fields you want, you may choose from `['stock_code', 'trade_date', 'open', 'high',\
    'low', 'close', 'vol', 'amount', 'adj_factor']`, or enter `'*'`
    to choose all.
    begin_date: str, deafult `"1999-12-19"`
        The date youe want the query to begin.
    end_date: str, default `datetime.now().strftime("%Y-%d-%m")`
        The date you want the query to end.
    stock_code: Union[Literal["*"], List[str]], default `'*'`
        The code of the stocks you want to query, or enter `'*'` for all.
    local_dir: pathlib.Path, default `Path.home() / ".sifeng/parquet/kline_day/"`
        The local path you want to data to download to, please do not change under normal circumsta
    nces.
    n_jobs: int, default `1`
        The number of workers for fetching.
    """
    def fdownload(mend):
        filename = f"KLINE-DAY{mend.strftime('%Y%m')}.parquet"
        update = not (local_dir / filename).exists()
        if mend.year == datetime.now().year and mend.month == datetime.now().month:
            update = True
        if update:
            resp = requests.get(baseurl + filename)
            with open(local_dir / filename, "wb") as file:
                file.write(resp.content)
    begin_date, end_date = pd.to_datetime(begin_date), pd.to_datetime(end_date)
    tasks = tqdm([delayed(fdownload)(_) for _ in pd.date_range(begin_date, end_date + MonthEnd(0), freq="M")], desc="Checking", unit="month", unit_scale=True, leave=False)
    local_dir.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=n_jobs, verbose=0)(tasks)
    if fields == "*":
        fields = ['stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'adj_factor']
    if stock_code == "*":
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / '*.parquet'}') WHERE trade_date BETWEEN '{begin_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
    elif isinstance(stock_code, str):
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / '*.parquet'}') WHERE trade_date BETWEEN '{begin_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}' AND stock_code = '{stock_code}'"
    else:
        connector = "', '"
        sql = f"SELECT {', '.join(fields)} FROM read_parquet('{local_dir / '*.parquet'}') WHERE trade_date BETWEEN '{begin_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}' AND stock_code IN ('{connector.join(stock_code)}')"
    return duckdb.sql(sql).df()
