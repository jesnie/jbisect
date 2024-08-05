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

`jbisect` provides the function `bisect_seq` for searching sequences:

```python
from jbisect import bisect_seq

print(bisect_seq("011222355", "2"))
```

By default the entire sequence is searched, but you can use the parameters `low` and `high` to limit
the search range:

```python
print(bisect_seq("011222355", "2", low=1, high=5))
```

You can use the `side` parameters to configure whether to return the first match, or just past the
last match:

```python
print(bisect_seq("011222355", "2", side="right"))
```

If you have a sequence that is descending, instead of ascending, you need to set the `ordering`
parameter:

```python
print(bisect_seq("553222110", "2", ordering="descending"))
```

# Searching functions:

The functions `bisect_int_fn` and `bisect_float_fn` can be used to search a function instead of a
sequence. These functions take the same `low`, `high`, `side` and `ordering` arguments as
`bisect_seq`.

```python
from jbisect import bisect_int_fn, bisect_float_fn

print(bisect_int_fn(lambda i: i * i, 16))
print(bisect_float_fn(lambda i: i * i, 2.0))
```

# Searching predicates:

Finally the functions `bisect_int_pred` and `bisect_float_pred` can be used to find the first value
accepted by a predicate. `pred` must be a function that returns a `bool`, and for which there exists
some `x` so that for all `y<x` `pred(y)` is `False`; and for all `y>=x` `pred(y)` is
`True`. `bisect_*_pred` will then find `x`.

```python
from jbisect import bisect_int_pred, bisect_float_pred

print(bisect_int_pred(lambda i: i * i >= 16))
print(bisect_float_pred(lambda i: i * i >= 2.0))
```
