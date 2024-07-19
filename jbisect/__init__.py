# pylint: disable=unnecessary-lambda-assignment

from collections.abc import Sequence
from math import exp2, inf, log2, nextafter, sqrt
from sys import float_info
from typing import (
    Any,
    Callable,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    assert_never,
)

__version__ = "0.1.0"


_DEBUG = True


class SupportsLess(Protocol):
    def __lt__(self, __other: Self) -> bool: ...


N = TypeVar("N", int, float)
L = TypeVar("L", bound=SupportsLess)
Side: TypeAlias = Literal["left", "right"]
Ordering: TypeAlias = Literal["ascending", "descending"]


def make_pred(
    fn: Callable[[N], L], target: L, side: Side, ordering: Ordering
) -> Callable[[N], bool]:
    if ordering == "ascending":
        if side == "left":
            return lambda x: fn(x) < target
        elif side == "right":
            if hasattr(target, "__le__"):
                return lambda x: fn(x) <= target  # type: ignore[operator]
            else:
                return lambda x: (y := fn(x)) < target or y == target
        else:
            assert_never(side)
    elif ordering == "descending":
        if side == "left":
            return lambda x: target < fn(x)
        elif side == "right":
            if hasattr(target, "__le__"):
                return lambda x: target <= fn(x)
            else:
                return lambda x: target < (y := fn(x)) or target == y
        else:
            assert_never(side)


def bisect_seq(
    seq: Sequence[L],
    target: L,
    *,
    low: int,
    high: int,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    return bisect_int_fn(lambda i: seq[i], target, low=low, high=high, side=side, ordering=ordering)


def bisect_int_fn(
    fn: Callable[[int], L],
    target: L,
    *,
    low: int,
    high: int,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    print(fn, target, low, high, side, ordering)
    return bisect_int_bool_fn(
        make_pred(fn, target, side, ordering), low=low, high=high, ordering="ascending"
    )


def bisect_int_bool_fn(
    pred: Callable[[int], bool],
    *,
    low: int,
    high: int,
    ordering: Ordering = "ascending",
) -> int:
    assert low <= high, (low, high)
    print(pred, low, high, ordering)

    if ordering == "descending":
        pred_ = pred
        pred = lambda i: not pred_(i)

    if low == high:
        print("No interval", low, high)
        return low
    if not pred(low):
        print("No valid interval", low, pred(low))
        return low
    if pred(high - 1):
        print("No invalid interval", high, pred(high - 1))
        return high

    while True:
        mid = (low + high) // 2
        if _DEBUG:
            assert low <= mid < high
            print("low", low, "high", high, "mid", mid)
        if mid == low:
            break
        if pred(mid):
            low = mid
        else:
            high = mid

    return high


def bisect_float_fn(
    fn: Callable[[float], L],
    target: L,
    *,
    low: float | None = None,
    high: float | None = None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> float:
    return bisect_float_bool_fn(
        make_pred(fn, target, side, ordering), low=low, high=high, ordering="ascending"
    )


def _float_mid(low: float, high: float) -> float:
    if low < 0.0 < high:
        return 0.0
    negative = False
    if low < 0.0:
        assert low < high <= 0.0, (low, high)
        negative = True
        low, high = -high, -low
    assert 0.0 <= low < high, (low, high)
    log_low = log2(low if low != 0.0 else float_info.min)
    log_high = log2(high)
    if log_high - log_low > 1:
        log_mid = log_low + (log_high - log_low) / 2
        mid = exp2(log_mid)
    else:
        mid = low + (high - low) / 2
    return -mid if negative else mid


def bisect_float_bool_fn(
    pred: Callable[[float], bool],
    *,
    low: float | None = None,
    high: float | None = None,
    ordering: Ordering = "ascending",
) -> float:
    if low is None:
        low = -float_info.max
    if high is None:
        high = float_info.max
    assert low <= high, (low, high)

    print("bisect_float_bool_fn", low, high, ordering)

    if ordering == "descending":
        pred_ = pred
        pred = lambda i: not pred_(i)

    if low == high:
        print("No interval")
        return low
    if not pred(low):
        print("No valid interval")
        return low
    if pred(nextafter(high, -inf)):
        print("No invalid interval")
        return high

    while True:
        mid = _float_mid(low, high)
        if mid == high:  # Deal with rounding up...
            mid = nextafter(mid, -inf)
        if _DEBUG:
            print("low", low, "high", high, "mid", mid)
        if mid == low:
            break
        if pred(mid):
            low = mid
        else:
            high = mid

    return high
