# pylint: disable=unnecessary-lambda-assignment

from collections.abc import Sequence
from math import exp2, inf, log2, nextafter
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


def _make_pred(
    fn: Callable[[N], L], target: L, side: Side, ordering: Ordering
) -> Callable[[N], bool]:
    if ordering == "ascending":
        if side == "left":
            if hasattr(target, "__le__"):
                return lambda x: target <= fn(x)
            else:
                return lambda x: target < (y := fn(x)) or target == y
        elif side == "right":
            return lambda x: target < fn(x)
        else:
            assert_never(side)
    elif ordering == "descending":
        if side == "left":
            if hasattr(target, "__le__"):
                return lambda x: fn(x) <= target  # type: ignore[operator]
            else:
                return lambda x: (y := fn(x)) < target or y == target
        elif side == "right":
            return lambda x: fn(x) < target
        else:
            assert_never(side)
    else:
        assert_never(ordering)


def bisect_seq(
    seq: Sequence[L],
    target: L,
    *,
    low: int | None = None,
    high: int | None = None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    """
    Binary search on a sorted `Sequence`. Returns an index where `target` should be inserted to
    maintain the ordering.

    :param seq: Sequence to search.
    :param target: The value to search for.
    :param low: Lower limit on the indices to search. If `None` will be set to `0`.
    :param high: Upper limit on the indices to search. If `None` will be set to the length of the
        sequence.
    :param side: If "left", returns the lowest possible index to insert `target` to maintain
        ordering. If `right` returns the highest possible index.
    :param ordering: Whether the sequence is sorted "ascending" or "descending".
    """
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
    low: int | None = None,
    high: int | None = None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> int:
    """
    Binary search on a monotonic function that takes an integer argument.

    :param fn: Function to search.
    :param target: The value to search for.
    :param low: If set defines the lowest possible input argument to search. If set, the function
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound - this may loop forever if no input argument is small enough to find `target`.
    :param high: If set defines the highest possible input argument to search. If set, the function
        will *not* be called with this argument, though this value will be returned if no lower
        argument value produces the `target` value. If unset, an exponential search is performed for
        the lower bound - this may loop forever if no input argument is big enough to find `target`.
    :param side: If "left", returns the lowest argument value that produces a value greater than or
        equal to `target`. If "right" returns the lowest argument value that produces a value
        strictly greater than `target`.
    :param ordering: Whether the function outputs are "ascending" or "descending".
    """
    return bisect_int_pred(
        _make_pred(fn, target, side, ordering),
        low=low,
        high=high,
    )


def bisect_int_pred(
    pred: Callable[[int], bool],
    *,
    low: int | None = None,
    high: int | None = None,
) -> int:
    """
    Binary search on a predicate that takes an integer argument.

    Assume there exists some `i`, so that for all `j<i` `pred(j)` is `False` and for all `j>=i`
    `pred(j)` is `True`. This function uses binary search to find this `i`.

    :param pred: Predicate to search.
    :param low: If set defines the lowest possible input argument to search. If set, the predicate
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound - this may loop forever if no input argument is small enough to be valid.
    :param high: If set defines the highest possible input argument to search. If set, the predicate
        will *not* be called with this argument, though this value will be returned if no lower
        argument value is valid. If unset, an exponential search is performed for
        the lower bound - this may loop forever if no input argument is big enough to be valid.
    """
    return _bisect_num_pred(
        pred,
        _int_suggest,
        low=low,
        high=high,
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
    """
    Binary search on a monotonic function that takes a floating-point argument.

    :param fn: Function to search.
    :param target: The value to search for.
    :param low: If set defines the lowest possible input argument to search. If set, the function
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound - this may loop forever if no input argument is small enough to find `target`.
    :param high: If set defines the highest possible input argument to search. If set, the function
        will *not* be called with this argument, though this value will be returned if no lower
        argument value produces the `target` value. If unset, an exponential search is performed for
        the lower bound - this may loop forever if no input argument is big enough to find `target`.
    :param side: If "left", returns the lowest argument value that produces a value greater than or
        equal to `target`. If "right" returns the lowest argument value that produces a value
        strictly greater than `target`.
    :param ordering: Whether the function outputs are "ascending" or "descending".
    """
    return bisect_float_pred(
        _make_pred(fn, target, side, ordering),
        low=low,
        high=high,
    )


def bisect_float_pred(
    pred: Callable[[float], bool],
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    """
    Binary search on a predicate that takes an floating-point argument.

    Assume there exists some `i`, so that for all `j<i` `pred(j)` is `False` and for all `j>=i`
    `pred(j)` is `True`. This function uses binary search to find this `i`.

    :param pred: Predicate to search.
    :param low: If set defines the lowest possible input argument to search. If set, the predicate
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound - this may loop forever if no input argument is small enough to be valid.
    :param high: If set defines the highest possible input argument to search. If set, the predicate
        will *not* be called with this argument, though this value will be returned if no lower
        argument value is valid. If unset, an exponential search is performed for
        the lower bound - this may loop forever if no input argument is big enough to be valid.
    """
    return _bisect_num_pred(
        pred,
        _float_suggest,
        low=low,
        high=high,
    )


def _bisect_num_pred(
    pred: Callable[[N], bool],
    suggest: Callable[[N | None, N | None], N],
    *,
    low: N | None,
    high: N | None,
) -> N:
    if low is not None and high is not None:
        assert low <= high, (low, high)
        if low == high:
            return low

    if low is not None and pred(low):
        return low

    if high is not None and not pred(prev_num(high)):
        return high

    while True:
        mid = suggest(low, high)
        if mid == low:
            break
        if not pred(mid):
            low = mid
        else:
            high = mid

    assert high is not None
    return high
