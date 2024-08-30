# jbisect

Reusable implementation of the bisect / binary search algorithm.

This (obviously) competes with the standard library
[`bisect`](https://docs.python.org/3.12/library/bisect.html#module-bisect) package. Whereas `bisect`
only searches lists this package supports searching on a function, and supports both integer and
floating-point indices.

## Install with:

```bash
pip install jbisect
```

## Basic searching

`jbisect` provides the function `search_seq` for searching sequences:

```python
from jbisect import search_seq

print(search_seq("011222355", "2"))
```

By default the entire sequence is searched, but you can use the parameters `low` and `high` to limit
the search range:

```python
print(search_seq("011222355", "2", low=1, high=5))
```

You can use the `side` parameters to configure whether to return the first match, or just past the
last match:

```python
print(search_seq("011222355", "2", side="right"))
```

If you have a sequence that is descending, instead of ascending, you need to set the `ordering`
parameter:

```python
print(search_seq("553222110", "2", ordering="descending"))
```

## Searching functions:

The functions `search_int_fn` and `search_float_fn` can be used to search a function instead of a
sequence. These functions take the same `low`, `high`, `side` and `ordering` arguments as
`search_seq`.

```python
from jbisect import search_int_fn, search_float_fn

print(search_int_fn(lambda i: i * i, 16, low=0))
print(search_float_fn(lambda i: i * i, 2.0, low=0.0))
```

## Searching predicates:

Finally the functions `search_int_pred` and `search_float_pred` can be used to find the first value
accepted by a predicate. `pred` must be a function that returns a `bool`, and for which there exists
some `x` so that for all `y<x` `pred(y)` is `False`; and for all `y>=x` `pred(y)` is
`True`. `search_*_pred` will then find `x`.

```python
from jbisect import search_int_pred, search_float_pred

print(search_int_pred(lambda i: i * i >= 16, low=0))
print(search_float_pred(lambda i: i * i >= 2.0, low=0.0))
```

## NumPy support:

The package `jbisect.numpy` adds functionality for searching NumPy arrays and functions.
Again, this obviously competes with the NumPy function
[`searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html).
However, `searchsorted` is limited in that it only searches existing arrays, and not functions.

`jbisect.numpy` provides three functions `search_numpy_array`, `search_numpy_fn` and
`search_numpy_pred`, mirroring the pure-python functions above. In the case of NumPy, we do not need
to distinguish between `int` and `float` up-front, as this is determined by the `dtype`.

`search_numpy_array` takes the new argument `axis` that determines which axis of the input array to
search.

`search_numpy_fn` and `search_numpy_pred` takes the new arguments `dtype` and `shape` which
determines the dtype and shape of the input to the function/predicate, and the return type of the
function.

```python
import numpy as np

from jbisect.numpy import search_numpy_array, search_numpy_fn, search_numpy_pred

print(search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2))

print(
    search_numpy_array(
        [
            [
                [111, 112, 113, 114],
                [121, 122, 123, 124],
                [131, 132, 133, 134],
            ],
            [
                [211, 212, 213, 214],
                [221, 222, 223, 224],
                [231, 232, 233, 234],
            ],
        ],
        [[122], [223]],
        axis=1,
    )
)

print(search_numpy_fn(lambda i: i * i, 16, low=0, high=1000, shape=(), dtype=np.int64))

print(search_numpy_pred(lambda i: i >= 4, shape=[], dtype=np.int64))
```