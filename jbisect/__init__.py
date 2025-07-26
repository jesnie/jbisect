# pylint: disable=unnecessary-lambda-assignment

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from math import exp2, inf, log2, nextafter
from sys import float_info
from typing import (
    Literal,
    Protocol,
    Self,
    TypeVar,
    assert_never,
)

__version__ = "0.1.0"


class SupportsLess(Protocol):
    def __lt__(self, /, other: Self) -> bool: ...


N = TypeVar("N", int, float)
L = TypeVar("L", bound=SupportsLess)
type Side = Literal["left", "right"]
type Ordering = Literal["ascending", "descending"]


class _Operations[N: (int, float)](ABC):
    def __init__(self) -> None:
        self.min_value: N | None
        self.max_value: N | None
        self.suggest_no_low_high: N

    @abstractmethod
    def suggest_no_low(self, high: N) -> N: ...

    @abstractmethod
    def suggest_no_high(self, low: N) -> N: ...

    @abstractmethod
    def suggest(self, low: N, high: N) -> N: ...

    @abstractmethod
    def prev(self, x: N) -> N: ...


class _IntegerOperations(_Operations[int]):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.min_value = None
        self.max_value = None
        self.suggest_no_low_high = 0

    def suggest_no_low(self, high: int) -> int:
        return min(2 * high, -16)

    def suggest_no_high(self, low: int) -> int:
        return max(2 * low, 16)

    def suggest(self, low: int, high: int) -> int:
        return (low + high) // 2

    def prev(self, x: int) -> int:
        return x - 1


_INT_OPERATIONS = _IntegerOperations()


class _FloatOperations(_Operations[float]):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.min_value: float = -float_info.max
        self.max_value: float = float_info.max
        self.suggest_no_low_high: float = 0.0

    def suggest_no_low(self, high: float) -> float:
        if high > 0.0:
            return 0.0
        if high > -2.0:
            return -2.0
        try:
            return -(high**2)
        except OverflowError:
            return -float_info.max

    def suggest_no_high(self, low: float) -> float:
        if low < 0.0:
            return 0.0
        if low < 2.0:
            return 2.0
        try:
            return low**2
        except OverflowError:
            return float_info.max

    def _mid(self, x: float, y: float) -> float:
        return x + (y - x) / 2

    def _nonnegative_suggest(self, low: float, high: float) -> float:
        assert 0.0 <= low < high, (low, high)

        log_low = log2(low if low != 0.0 else float_info.min)
        log_high = log2(high)
        if log_high - log_low > 1:
            return exp2(self._mid(log_low, log_high))

        return self._mid(low, high)

    def suggest(self, low: float, high: float) -> float:
        if low < 0.0 < high:
            return 0.0

        mid = (
            -self._nonnegative_suggest(-high, -low)
            if low < 0
            else self._nonnegative_suggest(low, high)
        )

        if mid == high:  # Deal with rounding up...
            mid = self.prev(mid)

        return mid

    def prev(self, x: float) -> float:
        return nextafter(x, -inf)


_FLOAT_OPERATIONS = _FloatOperations()


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


def search_seq(
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

    return search_int_fn(
        lambda i: seq[i],
        target,
        low=low,
        high=high,
        side=side,
        ordering=ordering,
    )


def search_int_fn(
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
    return search_int_pred(
        _make_pred(fn, target, side, ordering),
        low=low,
        high=high,
    )


def search_int_pred(
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
    return _search_pred(
        _INT_OPERATIONS,
        pred,
        low=low,
        high=high,
    )


def search_float_fn(
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
    return search_float_pred(
        _make_pred(fn, target, side, ordering),
        low=low,
        high=high,
    )


def search_float_pred(
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
    return _search_pred(
        _FLOAT_OPERATIONS,
        pred,
        low=low,
        high=high,
    )


def _search_pred(
    ops: _Operations[N],
    pred: Callable[[N], bool],
    *,
    low: N | None,
    high: N | None,
) -> N:
    if low is not None and high is not None:
        assert low <= high, (low, high)
        if low == high:
            return low

    low_ = low
    if low_ is None and ops.min_value is not None:
        low_ = ops.min_value
    if low_ is not None and pred(low_):
        return low_

    high_ = high
    if high_ is None and ops.max_value is not None:
        high_ = ops.max_value
    if high_ is not None and not pred(ops.prev(high_)):
        return high_

    while True:
        if low is None:
            mid = ops.suggest_no_low_high if high is None else ops.suggest_no_low(high)
        elif high is None:
            mid = ops.suggest_no_high(low)
        else:
            mid = ops.suggest(low, high)

        if mid == low:
            break
        if not pred(mid):
            low = mid
        else:
            high = mid

    assert high is not None
    return high
