from math import inf, nextafter

from jbisect import (
    search_float_fn,
    search_float_pred,
    search_int_fn,
    search_int_pred,
    search_seq,
)


def test_search_int_pred() -> None:
    assert search_int_pred(lambda i: i >= 4) == 4

    assert search_int_pred(lambda i: i >= 4, low=4) == 4
    assert search_int_pred(lambda i: i >= 4, low=5) == 5

    assert search_int_pred(lambda i: i >= 4, high=4) == 4
    assert search_int_pred(lambda i: i >= 4, high=3) == 3


def test_search_int_fn() -> None:
    assert search_int_fn(lambda i: i * i, 16, low=0) == 4

    assert search_int_fn(lambda i: i * i, 16, low=4) == 4
    assert search_int_fn(lambda i: i * i, 16, low=5) == 5

    assert search_int_fn(lambda i: i * i, 16, low=0, high=4) == 4
    assert search_int_fn(lambda i: i * i, 16, low=0, high=3) == 3

    assert search_int_fn(lambda i: i * i, 16, low=0, side="right") == 5

    assert search_int_fn(lambda i: -i * i, -16, low=0, ordering="descending") == 4


def test_search_seq() -> None:
    assert search_seq("", "2") == 0
    assert search_seq("2", "2") == 0
    assert search_seq("011222355", "2") == 3

    assert search_seq("011222355", "2", low=3) == 3
    assert search_seq("011222355", "2", low=4) == 4

    assert search_seq("011222355", "2", high=3) == 3
    assert search_seq("011222355", "2", high=2) == 2

    assert search_seq("011222355", "2", side="right") == 6

    assert search_seq("553222110", "2", ordering="descending") == 3


def test_search_float_pred() -> None:
    assert search_float_pred(lambda i: i >= 4.0) == 4.0

    assert search_float_pred(lambda i: i >= 4.0, low=4.0) == 4.0
    assert search_float_pred(lambda i: i >= 4.0, low=5.0) == 5.0

    assert search_float_pred(lambda i: i >= 4.0, high=4.0) == 4.0
    assert search_float_pred(lambda i: i >= 4.0, high=3.0) == 3.0


def test_search_float_fn() -> None:
    assert search_float_fn(lambda i: i * i, 16.0, low=0.0) == 4.0

    assert search_float_fn(lambda i: i * i, 16.0, low=4.0) == 4.0
    assert search_float_fn(lambda i: i * i, 16.0, low=5.0) == 5.0

    assert search_float_fn(lambda i: i * i, 16.0, low=0.0, high=4.0) == 4.0
    assert search_float_fn(lambda i: i * i, 16.0, low=0.0, high=3.0) == 3.0

    assert search_float_fn(lambda i: i * i, 16.0, low=0.0, side="right") == nextafter(4.0, inf)

    assert search_float_fn(lambda i: -i * i, -16.0, low=0.0, ordering="descending") == 4.0
