from jbisect import (
    search_float_fn,
    search_float_pred,
    search_int_fn,
    search_int_pred,
    search_seq,
)

print(search_seq("011222355", "2"))
print(search_seq("011222355", "2", low=1, high=5))
print(search_seq("011222355", "2", side="right"))
print(search_seq("553222110", "2", ordering="descending"))

print(search_int_fn(lambda i: i * i, 16))

print(search_int_pred(lambda i: i * i >= 16))

print(search_float_fn(lambda i: i * i, 2.0))

print(search_float_pred(lambda i: i * i >= 2.0))
