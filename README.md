# pororo-ocr
`pororo-ocr` inspired by [kakaobrain/pororo](https://github.com/kakaobrain/pororo), rebuilding only the OCR task of the pororo package for lighteweight and fast performance.

## Installation

- `pororo-ocr` is based on `torch>=1.6(cuda 10.1)` and `python>=3.6`

- You can install a package through the command below:

```console
pip install pororo-ocr
```

## Usage

- `pororo-ocr` can be used as follows:
- First, in order to import `pororo-ocr`, you must execute the following snippet

```python
>>> import prrocr
```

- After the import, you can check languages currently supported by the `pororo-ocr` through the following commands

```python
>>> import prrocr
>>> prrocr.ocr.get_available_langs()
"Available lanugages are ['en', 'ko']"
```

- To check which models are supported by each task, you can go through the following process

```python
>>> import prrocr
>>> prrocr.ocr.get_available_models()
"Available models are {'en': ['brainocr'], 'ko': ['brainocr']}"
```

- If you want to perform in specific language, you can put the language name in the `lang` argument

```python
>>> import prrocr
>>> ocr = prrocr.ocr(lang="en")
```

- After object construction, it can be used in a way that passes the input value as follows:

```python
>>> ocr("sample.jpg")
['MAKE TODAY', 'TOLERABLE']
```

- If you want to get position information for each string, you can turn on `detail` argument as follows:

```python
>>> ocr("sample.jpg", detail=True)
{'description': ['MAKE TODAY', 'TOLERABLE'], 'bounding_poly': [{'description': 'MAKE TODAY', 'vertices': [{'x': 585, 'y': 397}, {'x': 730, 'y': 397}, {'x': 730, 'y': 520}, {'x': 585, 'y': 520}]}, {'description': 'TOLERABLE', 'vertices': [{'x': 588, 'y': 558}, {'x': 884, 'y': 558}, {'x': 884, 'y': 612}, {'x': 588, 'y': 612}]}]}
```


## License

`pororo-ocr` project is licensed under the terms of **the Apache License 2.0**.

