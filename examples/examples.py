from jbisect import (
    bisect_float_fn,
    bisect_float_pred,
    bisect_int_fn,
    bisect_int_pred,
    bisect_seq,
)

print(bisect_seq("011222355", "2"))
print(bisect_seq("011222355", "2", low=1, high=5))
print(bisect_seq("011222355", "2", side="right"))
print(bisect_seq("553222110", "2", ordering="descending"))

print(bisect_int_fn(lambda i: i * i, 16))

print(bisect_int_pred(lambda i: i * i >= 16))

print(bisect_float_fn(lambda i: i * i, 2.0))

print(bisect_float_pred(lambda i: i * i >= 2.0))
