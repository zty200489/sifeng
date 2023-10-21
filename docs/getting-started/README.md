# Getting started with Sifeng

## Installation

It is recommended that you install using **`pip`**, and make sure to install the latest possible version to aviod any problems:

```shell
$ pip install sifeng            # normal install
$ pip install --upgrade sifeng  # upgrade if needed
```

Alternatively, it is also possible to install sifeng by cloning this repository:

```shell
$ git clone https://github.com/zty200489/sifeng.git
$ cd sifeng
$ pip install .
```

sifeng requires the following dependencies to work:

- [Pandas](https://pandas.pydata.org/)
- [Duckdb](https://duckdb.org/)
- [joblib](https://pypi.org/project/joblib/)
- [requests](https://github.com/psf/requests)
- [tqdm](https://tqdm.github.io/)

Please make sure they are properly installed, or simply use **pip** for your convenience.

## Fetching data

For example, you can fetch kline (HLOC) data from remote server with the `sifeng.fetcher.kline_day` function.

```python
from sifeng.fetcher import kline_day

df = kline_day(fields=["trade_date", "stock_code", "close", "adj_factor"],
               begin_date="2023-01-01",
               end_date="2023-01-31")
```

to fetch the close prices of the Chinese A-share stocks between `2023-01-01` and `2023-01-31`, along with the adjusting factor. For more information, you can check [API Reference](../api-reference/README)
