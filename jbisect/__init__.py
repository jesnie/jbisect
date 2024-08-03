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


def prev_int(i: int) -> int:
    return i - 1


def prev_float(x: float) -> float:
    return nextafter(x, -inf)


def prev_num(x: N) -> N:
    return prev_int(x) if isinstance(x, int) else prev_float(x)


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
    low: int | None,
    high: int | None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    if low is None:
        low = 0
    if high is None:
        high = len(seq)
    assert 0 <= low <= high <= len(seq), (low, high, len(seq))
    print("bisect_seq", seq, target, low, high, side, ordering)
    return bisect_int_fn(lambda i: seq[i], target, low=low, high=high, side=side, ordering=ordering)


def bisect_int_fn(
    fn: Callable[[int], L],
    target: L,
    *,
    low: int | None,
    high: int | None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    print(fn, target, low, high, side, ordering)
    return bisect_int_bool_fn(
        make_pred(fn, target, side, ordering), low=low, high=high, ordering="ascending"
    )


def _int_mid(low: int | None, high: int | None) -> int:
    if low is None:
        if high is None:
            return 0
        return min(2 * high, -16)
    if high is None:
        return max(2 * low, 16)
    return (low + high) // 2


def bisect_int_bool_fn(
    pred: Callable[[int], bool],
    *,
    low: int | None,
    high: int | None,
    ordering: Ordering = "ascending",
) -> int:

    if ordering == "descending":
        pred_ = pred
        pred = lambda i: not pred_(i)

    print("bisect_int_bool_fn", pred, low, high, ordering)

    if low is not None:
        if high is not None:
            assert low <= high, (low, high)
            if low == high:
                print("No interval", low, high)
                return low

        if not pred(low):
            print("No valid interval", low, pred(low))
            return low

    if high is not None:
        if pred(prev_int(high)):
            print("No invalid interval", high, pred(prev_int(high)))
            return high

    while True:
        mid = _int_mid(low, high)
        if _DEBUG:
            print("low", low, "high", high, "mid", mid)
        if mid == low:
            break
        if pred(mid):
            low = mid
        else:
            high = mid

    assert high is not None
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
    if pred(prev_float(high)):
        print("No invalid interval")
        return high

    while True:
        mid = _float_mid(low, high)
        if mid == high:  # Deal with rounding up...
            mid = prev_float(high)
        if _DEBUG:
            print("low", low, "high", high, "mid", mid)
        if mid == low:
            break
        if pred(mid):
            low = mid
        else:
            high = mid

    return high
