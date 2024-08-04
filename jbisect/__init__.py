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


def _int_suggest(low: int | None, high: int | None) -> int:
    if low is None:
        if high is None:
            return 0
        return min(2 * high, -16)
    if high is None:
        return max(2 * low, 16)
    return (low + high) // 2


def _nonnegative_float_suggest(low: float, high: float) -> float:
    assert 0.0 <= low < high, (low, high)

    log_low = log2(low if low != 0.0 else float_info.min)
    log_high = log2(high)
    if log_high - log_low > 1:
        log_mid = log_low + (log_high - log_low) / 2
        return exp2(log_mid)

    return low + (high - low) / 2


def _float_suggest(low: float | None, high: float | None) -> float:
    if low is None:
        low = -float_info.max
    if high is None:
        high = float_info.max

    if low < 0.0 < high:
        return 0.0

    mid = (
        -_nonnegative_float_suggest(-high, -low)
        if low < 0
        else _nonnegative_float_suggest(low, high)
    )

    if mid == high:  # Deal with rounding up...
        mid = prev_float(mid)

    return mid


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

    return bisect_int_fn(
        lambda i: seq[i],
        target,
        low=low,
        high=high,
        side=side,
        ordering=ordering,
    )


def bisect_int_fn(
    fn: Callable[[int], L],
    target: L,
    *,
    low: int | None,
    high: int | None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    return bisect_int_bool_fn(
        make_pred(fn, target, side, ordering),
        low=low,
        high=high,
        ordering="ascending",
    )


def bisect_int_bool_fn(
    pred: Callable[[int], bool],
    *,
    low: int | None,
    high: int | None,
    ordering: Ordering = "ascending",
) -> int:
    return bisect_num_bool_fn(
        pred,
        _int_suggest,
        low=low,
        high=high,
        ordering=ordering,
    )


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
        make_pred(fn, target, side, ordering),
        low=low,
        high=high,
        ordering="ascending",
    )


def bisect_float_bool_fn(
    pred: Callable[[float], bool],
    *,
    low: float | None = None,
    high: float | None = None,
    ordering: Ordering = "ascending",
) -> float:
    return bisect_num_bool_fn(
        pred,
        _float_suggest,
        low=low,
        high=high,
        ordering=ordering,
    )


def bisect_num_bool_fn(
    pred: Callable[[N], bool],
    suggest: Callable[[N | None, N | None], N],
    *,
    low: N | None,
    high: N | None,
    ordering: Ordering = "ascending",
) -> N:
    if ordering == "descending":
        pred_ = pred
        pred = lambda i: not pred_(i)

    if low is not None and high is not None:
        assert low <= high, (low, high)
        if low == high:
            return low

    if low is not None and not pred(low):
        return low

    if high is not None and pred(prev_num(high)):
        return high

    while True:
        mid = suggest(low, high)
        if mid == low:
            break
        if pred(mid):
            low = mid
        else:
            high = mid

    assert high is not None
    return high
