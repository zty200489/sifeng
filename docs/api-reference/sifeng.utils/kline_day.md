```python
sifeng.utils.kline_day(fields = "*",
                       begin_date = "1999-12-19",
                       end_date = datetime.now().strftime("%Y-%d-%m"),
                       stock_code = "*",
                       local_dir = Path.home() / ".sifeng/parquet/kline_day/",
                       n_jobs = 1) -> pandas.DataFrame:
```

# Description

Returns a DataFrame that contains the kline (HLOC) data of Chinese A-share stocks.

# Parameters

- fields (`str`, `List[str]`, optional) - Specifies the fields you want to retrieve. It can be any sub-set (sub-list?) of `"stock_code"` | `"trade_date"` | `"open"` | `"high"` | `"low"` | `"close"` | `"vol"` | `"amount"` | `"adj_factor"`, or simply pass in `"*"` for the meaning of selecting all. Default: `"*"`.
  - `"stock_code"`: The code of the stock, in the format of `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`. eg. `"000001.SZ".
  - `"trade_date"`: The date of that entry, in the format of `"%Y-%m-%d"`. eg. `"2018-07-18"`.
  - `"open"`: The open price in the HLOC line.
  - `"high"`: The high price in the HLOC line.
  - `"low"`: The low price in the HLOC line.
  - `"close"`: The close price in the HLOC line.
  - `"vol"`: The volume traded, in unit of shares.
  - `"amount"`: The amount traded, in unit of 1k yuan.
  - `"adj_factor"`: The adjustment factor for cross-comparision across a long-period of time.
- begin_date (`str`, optional) - The date of which you want the query to begin. Though any format that the pandas auto-cast allows should perform without a problem, it is recommended to use the same format as `"%Y-%m-%d"`. Default: `"1999-12-19"`
- end_date (`str`, optional) - The date of which you want the query to end. Though any format that the pandas auto-cast allows should perform without a problem, it is recommended to use the same format as `"%Y-%m-%d"`. Default: `datetime.now().strftime("%Y-%d-%m")` (The current date.)
- stock_code (`str`, `List[str]`, optional) - The list of stocks you want to query. It should use the same format as `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`, you may pass in a single stock, a list of stocks, or simply `"*"` for all. Default: `"*"`.
- local_dir (Path, optional) - The local directory you want the cached data to be downloads to. It downloads to your personal directory by default. It is recommemded that you do not change this setting. Default: `Path.home() / ".sifeng/parquet/kline_day/"`.
- n_jobs (int, optional)- The number of workers you want the downloading to have. Though in theory, more workers will be faster, that isn'y necessarily the case. It is also recommended that you do not change this setting. Default: `1`.

# Example

```python
>>> from sifeng.utils import kline_day
>>> kline_day(["stock_code", "trade_date", "close", "adj_factor"], begin_date="2020-01-01", end_date="2020-01-31", stock_code="600000.SH")
   stock_code  trade_date  close  adj_factor
0   600000.SH  2020-01-23  11.35      12.714
1   600000.SH  2020-01-22  11.77      12.714
2   600000.SH  2020-01-21  12.08      12.714
3   600000.SH  2020-01-20  12.25      12.714
4   600000.SH  2020-01-17  12.23      12.714
5   600000.SH  2020-01-16  12.20      12.714
6   600000.SH  2020-01-15  12.25      12.714
7   600000.SH  2020-01-14  12.43      12.714
8   600000.SH  2020-01-13  12.41      12.714
9   600000.SH  2020-01-10  12.39      12.714
10  600000.SH  2020-01-09  12.37      12.714
11  600000.SH  2020-01-08  12.32      12.714
12  600000.SH  2020-01-07  12.50      12.714
13  600000.SH  2020-01-06  12.46      12.714
14  600000.SH  2020-01-03  12.60      12.714
15  600000.SH  2020-01-02  12.47      12.714
```