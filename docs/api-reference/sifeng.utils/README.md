# sifeng.fetcher

## About

`sifeng.utils` holds functions that make programming and researching easier. Jobs such as fetching data from remote servers are already provided.

## Functions

|Name|Table Type|Description|
|:--:|:--:|:--|
|[kline_day](kline_day.md)|cross-sectional|Chinese A-share stock market kline (HLOC), day frequency data.|
|[indicator_day](indicator_day.md)|cross-sectional|Chinese A-share stock market indicators, day frequency data.|
|[baisc_info](basic_info.md)|static|The fundamentals of A-share stocks. eg. list-date, industry, name, etc.|

## Appendix: stock data

The stock market data has three kinds of tables: cross-sectional tables, static tables, and stream tables.

The **cross-sectional tables** are the most frequently used table, they are stored as a panel, where each row is uniquely indexed by `(stock_code, trade_month)`, `(stock_code, trade_date)`, or `(stock_code, trade_time)` depending on the frequency of the data.

The **static tables** are static info about each stock, such as the list-date, delist-date, industry and so on. They remain static over a long period and thus is uniquely indexed by `stock_code` alone. By default, they will be updated on a daily basis, and your program will fetch them to update local cache if and only if the local cache timestamp is at least one day old.

The **stream tables** are data streams, most frequently along time. Raw text/news data for example is the simplest form of stream table, as they are ususally hard to be categorized into a specific stock. They will be uniquely indexed by `trade_month`, `trade_date`, or `trade_time` depending on the frequency of the data.
