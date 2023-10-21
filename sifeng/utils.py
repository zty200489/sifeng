import duckdb
import requests
from pathlib import Path

from typing import Union, Literal, List

__all__ = [
    "basic_info",
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
