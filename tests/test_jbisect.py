from math import inf, nextafter

from jbisect import (
    bisect_float_fn,
    bisect_float_pred,
    bisect_int_fn,
    bisect_int_pred,
    bisect_seq,
)


def test_bisect_int_pred() -> None:
    assert bisect_int_pred(lambda i: i >= 4) == 4

    assert bisect_int_pred(lambda i: i >= 4, low=4) == 4
    assert bisect_int_pred(lambda i: i >= 4, low=5) == 5

    assert bisect_int_pred(lambda i: i >= 4, high=4) == 4
    assert bisect_int_pred(lambda i: i >= 4, high=3) == 3


def test_bisect_int_fn() -> None:
    assert bisect_int_fn(lambda i: i * i, 16) == 4

    assert bisect_int_fn(lambda i: i * i, 16, low=4) == 4
    assert bisect_int_fn(lambda i: i * i, 16, low=5) == 5

    assert bisect_int_fn(lambda i: i * i, 16, high=4) == 4
    assert bisect_int_fn(lambda i: i * i, 16, high=3) == 3

    assert bisect_int_fn(lambda i: i * i, 16, side="right") == 5

    assert bisect_int_fn(lambda i: -i * i, -16, ordering="descending") == 4


def test_bisect_seq() -> None:
    assert bisect_seq("", "2") == 0
    assert bisect_seq("2", "2") == 0
    assert bisect_seq("011222355", "2") == 3

    assert bisect_seq("011222355", "2", low=3) == 3
    assert bisect_seq("011222355", "2", low=4) == 4

    assert bisect_seq("011222355", "2", high=3) == 3
    assert bisect_seq("011222355", "2", high=2) == 2

    assert bisect_seq("011222355", "2", side="right") == 6

    assert bisect_seq("553222110", "2", ordering="descending") == 3


def test_bisect_float_pred() -> None:
    assert bisect_float_pred(lambda i: i >= 4.0) == 4.0

    assert bisect_float_pred(lambda i: i >= 4.0, low=4.0) == 4.0
    assert bisect_float_pred(lambda i: i >= 4.0, low=5.0) == 5.0

    assert bisect_float_pred(lambda i: i >= 4.0, high=4.0) == 4.0
    assert bisect_float_pred(lambda i: i >= 4.0, high=3.0) == 3.0


def test_bisect_float_fn() -> None:
    assert bisect_float_fn(lambda i: i * i, 16.0) == 4.0

    assert bisect_float_fn(lambda i: i * i, 16.0, low=4.0) == 4.0
    assert bisect_float_fn(lambda i: i * i, 16.0, low=5.0) == 5.0

    assert bisect_float_fn(lambda i: i * i, 16.0, high=4.0) == 4.0
    assert bisect_float_fn(lambda i: i * i, 16.0, high=3.0) == 3.0

    assert bisect_float_fn(lambda i: i * i, 16.0, side="right") == nextafter(4.0, inf)

    assert bisect_float_fn(lambda i: -i * i, -16.0, ordering="descending") == 4.0
