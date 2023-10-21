```python
def basic_info(fields: Union[Literal["*"], List[str]] = "*",
               stock_code: Union[Literal["*"], List[str]] = "*",
               local_dir: Path = Path.home() / ".sifeng/parquet/status_charts/",
               force_update: bool = False) -> pandas.DataFrame:
```

# Description

Returns a DataFrame that contains the fundamentals of Chinese A-share stocks.

# Parameters

- fields (`str`, `List[str]`, optional) - Specifies the fields you want to retrieve. It can be any sub-set (sub-list?) of `"stock_code"` | `"stock_name"` | `"area"` | `"industry"` | `"sector"` | `"list_status"` | `"list_date"` | `"st_flag"`, or simply pass in `"*"` for the meaning of selecting all. Default: `"*"`.
  - `"stock_code"`: The code of the stock, in the format of `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`. eg. `"000001.SZ".
  - `"stock_name"`: The name of the stock.
  - `"area"`: In which are is the company behind this stock located in.
  - `"industry"`: The industry of the company behind this stock.
  - `"sector"`: On which sector is this stock listed. `0` for 主板 (Main-Board Market); `1` for 创业板 (Growth Enterprise Market); `2` for 北交所 (Beijing SE); `3` for 科创板 (SSE STAR MARKET).
  - `"list_status"`: The status of the stock.
  - `"list_date"`: The date of when the stock was listed.
  - `"st_flag"`: Whether the stock is marked as **Special Treatment** (ST).
- stock_code (`str`, `List[str]`, optional) - The list of stocks you want to query. It should use the same format as `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`, you may pass in a single stock, a list of stocks, or simply `"*"` for all. Default: `"*"`.
- local_dir (Path, optional) - The local directory you want the cached data to be downloads to. It downloads to your personal directory by default. It is recommemded that you do not change this setting. Default: `Path.home() / ".sifeng/parquet/status_charts/"`.
- force_update (bool) - Whether to force update the cached data. Default: `False`.

# Example

```python
>>> from sifeng.utils import basic_info
>>> basic_info(["stock_code", "area", "sector", "list_status", "st_flag"])
     stock_code area sector list_status st_flag
0     000001.SZ   深圳      0           L       0
1     000002.SZ   深圳      0           L       0
2     000004.SZ   深圳      0           L       0
3     000005.SZ   深圳      0           L       1
4     000006.SZ   深圳      0           L       0
...         ...  ...    ...         ...     ...
5289  873576.BJ   陕西      2           L       0
5290  873593.BJ   江苏      2           L       0
5291  873665.BJ   江苏      2           L       0
5292  873726.BJ   江苏      2           L       0
5293  689009.SH   北京      3           L       0

[5294 rows x 5 columns]
```