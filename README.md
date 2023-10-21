<h1 align="center">
    Si Feng: The machine learning quant research framework
</h1>

<p align="center">
    <a href="#stars"><img alt="stars" src="https://img.shields.io/github/stars/zty200489/sifeng"></a>
    <a href="#watchers"><img alt="watchers" src="https://img.shields.io/github/watchers/zty200489/sifeng"></a>
    <a href="#forks"><img alt="forks" src="https://img.shields.io/github/forks/zty200489/sifeng"></a>
    <a href="#version"><img alt="version" src="https://img.shields.io/badge/version-0.3.1-74c365"></a>
</p>

**Documentation: [Getting Started](https://zty200489.github.io/sifeng/#/getting-started/README) | [API Reference](https://zty200489.github.io/sifeng/#/api-reference/README)**  
**Releases: [PyPI](https://pypi.org/project/sifeng) | [Source](https://github.com/zty200489/sifeng/tree/master/sifeng) | [Changelog](https://zty200489.github.io/sifeng/#/change-log/README)**

Sifeng (思风, imported as `sifeng`) is a quant research framework built by our team, which also share the same name Sifeng. It is a python package mainly designed to be used within our team but also welcomes public users. Feel free to tell us what features you want to add to this project. Currently it supports many features such as:
- :book: Detailed [wiki](https://zty200489.github.io/sifeng/#/), and annotations.
- :chart_with_upwards_trend: Accessible and easy to use function for retrieving structurized data.
- :zap: SOTA deep learning models, bayesian nerual networks, useful tools, and so on.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Highlights](#highlights)

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

sifeng requires the following dependencies:

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Duckdb](https://duckdb.org/)
- [joblib](https://pypi.org/project/joblib/)
- [requests](https://github.com/psf/requests)
- [tqdm](https://tqdm.github.io/)
- [pytorch](https://pytorch.org/)
- [colorama](https://github.com/tartley/colorama)

## Highlights

- :chart_with_upwards_trend: We understand that in quant resarch, data is the key. So we provide easy-to-use API for retrieving data from remote OSS server. We currently only supports daily HLOC data, indicators and basic information, but we will include other kinds of data shortly in the future.
- :handshake: As out deep-learning models are based on torch, we support uniform and consistent APIs that can be pieced easily into your eexisting projects to ease you the burden of repetitive hard work.
- :bookmark_tabs: Detailed and fully-customizeable logs that fits specifically for your project. You can easily design your own log format or print custom log messages by inheriting the `VerboseBase` class and passing it into an trainer.
- :abacus: Model is the second most important part when doing quant researches. So we provide SOTA deep-learning models including `MultiheadedSelfAttentionModule`, `MixtureOfExpertsBlock` and even Bayesian modules to help you conduct research across various kinds of data.
