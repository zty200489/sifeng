```python
sifeng.fetcher.indicator_day(fields = "*",
                             begin_date = "1999-12-19",
                             end_date = datetime.now().strftime("%Y-%d-%m"),
                             stock_code = "*",
                             local_dir = Path.home() / ".sifeng/parquet/indicator_day/",
                             n_jobs = 1) -> pandas.DataFrame:
```

# Description

Returns a DataFrame that contains the indicators data of Chinese A-share stocks.

# Parameters

- fields (`str`, `List[str]`, optional) - Specifies the fields you want to retrieve. It can be any sub-set (sub-list?) of `"stock_code"` | `"trade_date"` | `"turnover_rate"` | `"turnover_rate_free"` | `"volume_ratio"` | `"pe"` | `"pe_ttm"` | `"pb"` | `"ps"` | `"ps_ttm"` | `"dv_ratio"` | `"dv_ttm"` | `"total_share"` | `"float_share"` | `"free_share"` | `"total_mv"` | `"circ_mv"`, or simply pass in `"*"` for the meaning of selecting all. Default: `"*"`.
  - `"stock_code"`: The code of the stock, in the format of `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`. eg. `"000001.SZ".
  - `"trade_date"`: The date of that entry, in the format of `"%Y-%m-%d"`. eg. `"2018-07-18"`.
  - `"turnover_rate"`: The turnover rate of the stock.
  - `"turnover_rate_free"`: The turnover rate of the free-float stock.
  - `"volume_ratio"`: The minute-average volume of past five days devided by the minute-average volume today.
  - `"pe"`: The price-to-earnings ratio.
  - `"pe_ttm"`: The trailing twelve-month price-to-earnings ratio.
  - `"pb"`: The price-to-book ratio.
  - `"ps"`: The price-to-sales ratio.
  - `"ps_ttm"`: The trailing twelve-month price-to-sales ratio.
  - `"dv_ratio"`: The dividend yield devided by the current stock price.
  - `"dv_ttm"`: The trailing twelve-month dividend yield devided by the current stock price.
  - `"total_share"`: The total number of shares.
  - `"float_share"`: The number of float shares.
  - `"free_share"`: The number of free shares.
  - `"total_mv"`: The total market value of the stock.
  - `"circ_mv"`: The circulating market value of the stock.
- begin_date (`str`, optional) - The date of which you want the query to begin. Though any format that the pandas auto-cast allows should perform without a problem, it is recommended to use the same format as `"%Y-%m-%d"`. Default: `"1999-12-19"`
- end_date (`str`, optional) - The date of which you want the query to end. Though any format that the pandas auto-cast allows should perform without a problem, it is recommended to use the same format as `"%Y-%m-%d"`. Default: `datetime.now().strftime("%Y-%d-%m")` (The current date.)
- stock_code (`str`, `List[str]`, optional) - The list of stocks you want to query. It should use the same format as `"{STOCK CODE}.{STOCK EXCHANGE CODE}"`, you may pass in a single stock, a list of stocks, or simply `"*"` for all. Default: `"*"`.
- local_dir (Path, optional) - The local directory you want the cached data to be downloads to. It downloads to your personal directory by default. It is recommemded that you do not change this setting. Default: `Path.home() / ".sifeng/parquet/indicator_day/"`.
- n_jobs (int, optional)- The number of workers you want the downloading to have. Though in theory, more workers will be faster, that isn'y necessarily the case. It is also recommended that you do not change this setting. Default: `1`.

# Example

```python
>>> from sifeng.utils import indicator_day
>>> indicator_day(["stock_code", "trade_date", "turnover_rate", "circ_mv"], begin_date="2020-01-01", end_date="2020-01-31", stock_code="600000.SH")
   stock_code  trade_date  turnover_rate       circ_mv
0   600000.SH  2020-01-23         0.2723  3.189777e+07
1   600000.SH  2020-01-22         0.2780  3.307813e+07
2   600000.SH  2020-01-21         0.1131  3.394935e+07
3   600000.SH  2020-01-20         0.0845  3.442711e+07
4   600000.SH  2020-01-17         0.0626  3.437090e+07
5   600000.SH  2020-01-16         0.0799  3.428659e+07
6   600000.SH  2020-01-15         0.1135  3.442711e+07
7   600000.SH  2020-01-14         0.1067  3.493298e+07
8   600000.SH  2020-01-13         0.0737  3.487677e+07
9   600000.SH  2020-01-10         0.0652  3.482056e+07
10  600000.SH  2020-01-09         0.0931  3.476436e+07
11  600000.SH  2020-01-08         0.1254  3.462384e+07
12  600000.SH  2020-01-07         0.1011  3.512970e+07
13  600000.SH  2020-01-06         0.1459  3.501729e+07
14  600000.SH  2020-01-03         0.1353  3.541074e+07
15  600000.SH  2020-01-02         0.1837  3.504539e+07
```
