```python
sifeng.utils.stat(model: torch.nn.Module,
                  sort: Literal[":idx", "idx:", ":name", "name:", ":train", "train:", ":params", "params:", ":memory", "memory:"] = "train:",
                  ) -> str:
```

# Description

Returns the anaylsis (parameters, memory consumption, etc.) of a torch model as a string.

# Parameters

- model (`torch.nn.Module`) - The model to be analyzed.
- sort: (`str`, Optional) - How to sort the output, a leading comma means ascending and a trailing comma means descending, you may choose from `"idx"`, `"name"`, `"train"`, `"params"`, `"memory"`, default `"train:"`.

# Example

```python
>>> from sifeng.utils import stat
... from torch import nn
>>> class Model(nn.Module):
...     def __init__(self):
...         super(Model, self).__init__()
...         self.mlp = nn.Sequential(
...             nn.Linear(300, 200),
...             nn.ReLU(),
...             nn.Linear(200, 1),
...         )
...
...     def forward(self, x):
...         return self.mlp(x)
>>> model = Model()
... print(stat(model))
┌─────┬──────────────┬────────────────────────┬────────┬────────┬────────────┐
│ idx │     name     │         shape          │ train: │ params │   memory   │
├─────┼──────────────┼────────────────────────┼────────┼────────┼────────────┤
│   0 │ mlp.0.weight │ torch.Size([200, 300]) │ True   │ 60 K   │ 234.38 KiB │
│   1 │ mlp.0.bias   │ torch.Size([200])      │ True   │ 200    │ 800 Bytes  │
│   2 │ mlp.2.weight │ torch.Size([1, 200])   │ True   │ 200    │ 800 Bytes  │
│   3 │ mlp.2.bias   │ torch.Size([1])        │ True   │ 1      │ 4 Bytes    │
├─────┴──────────────┴────────────────────────┴────────┴────────┴────────────┤
│ Model params: 60.4010 K                                                    │
│ Model memory: 235.9414 KiB                                                 │
└────────────────────────────────────────────────────────────────────────────┘
```