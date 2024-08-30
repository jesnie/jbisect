from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import inf
from typing import Any, Callable, Generic, TypeVar, assert_never, cast

import numpy as np
import numpy.typing as npt

from jbisect import Ordering, Side

S = TypeVar("S", bound=tuple[int, ...])
GD = TypeVar("GD", bound=np.generic)
ND = TypeVar("ND", bound=np.number[Any])
ID = TypeVar("ID", bound=np.integer[Any])
SID = TypeVar("SID", bound=np.signedinteger[Any])
UID = TypeVar("UID", bound=np.unsignedinteger[Any])
FD = TypeVar("FD", bound=np.floating[Any])


def _cast_scalar(x: int | float | np.number[Any], dtype: np.dtype[ND]) -> ND:
    return cast(ND, np.asarray(x, dtype=dtype))


class _Operations(ABC, Generic[ND]):

    def __init__(self) -> None:
        self.min_value: ND
        self.max_value: ND
        self.suggest_no_low_high: ND

    @abstractmethod
    def suggest_no_low(
        self,
        high: np.ndarray[S, np.dtype[ND]],
    ) -> np.ndarray[S, np.dtype[ND]]: ...

    @abstractmethod
    def suggest_no_high(
        self,
        low: np.ndarray[S, np.dtype[ND]],
    ) -> np.ndarray[S, np.dtype[ND]]: ...

    @abstractmethod
    def suggest(
        self,
        low: np.ndarray[S, np.dtype[ND]],
        high: np.ndarray[S, np.dtype[ND]],
    ) -> np.ndarray[S, np.dtype[ND]]: ...

    @abstractmethod
    def prev(
        self,
        x: np.ndarray[S, np.dtype[ND]],
    ) -> np.ndarray[S, np.dtype[ND]]: ...


class _IntegerOperations(_Operations[ID], Generic[ID]):

    def __init__(
        self,
        dtype: np.dtype[ID],
    ) -> None:
        super().__init__()
        self._iinfo = np.iinfo(dtype)
        self.min_value = _cast_scalar(self._iinfo.min, dtype)
        self.max_value = _cast_scalar(self._iinfo.max, dtype)

    def prev(
        self,
        x: np.ndarray[S, np.dtype[ID]],
    ) -> np.ndarray[S, np.dtype[ID]]:
        return np.where(x == self.min_value, x, x - 1)


class _SignedIntegerOperations(_IntegerOperations[SID], Generic[SID]):

    def __init__(
        self,
        dtype: np.dtype[SID],
    ) -> None:
        super().__init__(dtype)
        self.suggest_no_low_high = _cast_scalar(0, dtype)

    def suggest_no_low(
        self,
        high: np.ndarray[S, np.dtype[SID]],
    ) -> np.ndarray[S, np.dtype[SID]]:
        return np.where(
            high <= self.min_value // 2,
            self.min_value,
            np.minimum(
                2 * high,
                -self._iinfo.bits,
            ),
        )

    def suggest_no_high(
        self,
        low: np.ndarray[S, np.dtype[SID]],
    ) -> np.ndarray[S, np.dtype[SID]]:
        return np.where(
            low > self.max_value // 2,
            self.max_value,
            np.maximum(
                2 * low,
                self._iinfo.bits,
            ),
        )

    def suggest(
        self,
        low: np.ndarray[S, np.dtype[SID]],
        high: np.ndarray[S, np.dtype[SID]],
    ) -> np.ndarray[S, np.dtype[SID]]:
        return np.where(
            (low < 0) & (0 <= high),
            (low + high) // 2,
            low + (high - low) // 2,
        )


class _UnsignedIntegerOperations(_IntegerOperations[UID], Generic[UID]):

    def __init__(
        self,
        dtype: np.dtype[UID],
    ) -> None:
        super().__init__(dtype)
        self.suggest_no_low_high = _cast_scalar(self._iinfo.bits, dtype)

    def suggest_no_low(
        self,
        high: np.ndarray[S, np.dtype[UID]],
    ) -> np.ndarray[S, np.dtype[UID]]:
        return cast(np.ndarray[S, np.dtype[UID]], high // 2)

    def suggest_no_high(
        self,
        low: np.ndarray[S, np.dtype[UID]],
    ) -> np.ndarray[S, np.dtype[UID]]:
        return np.where(
            low > self.max_value // 2,
            self.max_value,
            np.maximum(
                2 * low,
                self._iinfo.bits,
            ),
        )

    def suggest(
        self,
        low: np.ndarray[S, np.dtype[UID]],
        high: np.ndarray[S, np.dtype[UID]],
    ) -> np.ndarray[S, np.dtype[UID]]:
        return low + (high - low) // 2


class _FloatOperations(_Operations[FD], Generic[FD]):

    def __init__(
        self,
        dtype: np.dtype[FD],
    ) -> None:
        super().__init__()
        self._finfo = np.finfo(dtype)
        self.min_value = _cast_scalar(self._finfo.min, dtype)
        self.max_value = _cast_scalar(self._finfo.max, dtype)
        self.suggest_no_low_high = _cast_scalar(0.0, dtype)

    def suggest_no_low(
        self,
        high: np.ndarray[S, np.dtype[FD]],
    ) -> np.ndarray[S, np.dtype[FD]]:
        return np.maximum(np.where(high <= -2.0, -(high**2), -2.0), self._finfo.min)

    def suggest_no_high(
        self,
        low: np.ndarray[S, np.dtype[FD]],
    ) -> np.ndarray[S, np.dtype[FD]]:
        return np.minimum(np.where(low >= 2.0, low**2, 2.0), self._finfo.max)

    def _nonnegative_suggest(
        self,
        low: np.ndarray[S, np.dtype[FD]],
        high: np.ndarray[S, np.dtype[FD]],
    ) -> np.ndarray[S, np.dtype[FD]]:
        log_low = np.log2(np.where(low == 0.0, self._finfo.smallest_normal, low))
        log_high = np.log2(high)
        return np.where(
            log_high - log_low > 1,
            np.exp2(log_low + (log_high - log_low) / 2),
            low + (high - low) / 2,
        )

    def suggest(
        self,
        low: np.ndarray[S, np.dtype[FD]],
        high: np.ndarray[S, np.dtype[FD]],
    ) -> np.ndarray[S, np.dtype[FD]]:
        result = np.where(
            (low < 0.0) & (0.0 < high),
            0.0,
            np.where(
                low < 0.0,
                -self._nonnegative_suggest(-high, -low),
                self._nonnegative_suggest(low, high),
            ),
        )
        return np.where(result < high, result, np.nextafter(result, -inf))

    def prev(
        self,
        x: np.ndarray[S, np.dtype[FD]],
    ) -> np.ndarray[S, np.dtype[FD]]:
        return np.where(x == self.min_value, x, np.nextafter(x, -inf))


def _get_operations(dtype: np.dtype[ND]) -> _Operations[ND]:
    match dtype.kind:
        case "i":
            return _SignedIntegerOperations(dtype)  # type: ignore[type-var]
        case "u":
            return _UnsignedIntegerOperations(dtype)  # type: ignore[type-var]
        case "f":
            return _FloatOperations(dtype)  # type: ignore[type-var]
    raise ValueError(f"dtype {dtype} not supported.")


def search_numpy_array(
    arr: npt.ArrayLike,
    target: npt.ArrayLike,
    *,
    low: npt.ArrayLike | None = None,
    high: npt.ArrayLike | None = None,
    axis: int = 0,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> np.ndarray[S, np.dtype[np.uintp]]:
    """
    Binary search on a sorted NumPy "array", along one `axis`. Returns an index where `target`
    should be inserted to maintain the ordering.

    If `arr` has shape `(d0, d1, d2, ..., d_axis, ..., dn-2, dn-1, dn)`, the result will have shape
    `(d0, d1, d2, ..., dn-2, dn-1, dn)`; and `target`, `low` and `high` should broadcast to that
    shape.

    :param arr: Array to search.
    :param target: The value(s) to search for. Must broadcast to the result shape.
    :param low: Lower limit on the indices to search. If `None` will be set to `0`. Must broadcast
        to the result shape.
    :param high: Upper limit on the indices to search. If `None` will be set to the length of the
        axis we are searching. Must broadcast to the result shape.
    :param side: If "left", returns the lowest possible index to insert `target` to maintain
        ordering. If `right` returns the highest possible index.
    :param ordering: Whether the sequence is sorted "ascending" or "descending".
    """
    arr = np.asarray(arr)
    assert 0 <= axis < len(arr.shape)
    len_ = arr.shape[axis]
    shape = arr.shape[:axis] + arr.shape[axis + 1 :]
    dtype = np.uintp

    if len_ == 0:
        return np.broadcast_to(np.asarray(0, dtype=dtype), shape)

    if low is None:
        low = 0
    if high is None:
        high = len_
    low = np.asarray(low, dtype=dtype)
    high = np.asarray(high, dtype=dtype)

    assert (0 <= low <= high <= len_).all(), (low, high, len_)

    ranges = tuple(
        np.arange(l).reshape((1,) * i + (l,) + (1,) * (len(shape) - 1 - i))
        for i, l in enumerate(shape)
    )

    def fn(i: np.ndarray[S, np.dtype[np.uintp]]) -> np.ndarray[S, Any]:
        # TODO: Is there a simpler / more elegant way to do this?
        result = arr[ranges[:axis] + (i,) + ranges[axis:]]
        return result  # type: ignore[no-any-return]

    return search_numpy_fn(
        fn,
        target,
        low=low,
        high=high,
        shape=shape,
        dtype=dtype,
        side=side,
        ordering=ordering,
    )


def search_numpy_fn(
    fn: Callable[[np.ndarray[S, np.dtype[ND]]], np.ndarray[S, np.dtype[GD]]],
    target: npt.ArrayLike,
    *,
    low: npt.ArrayLike | None = None,
    high: npt.ArrayLike | None = None,
    shape: int | Iterable[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    side: Side = "left",
    ordering: Ordering = "ascending",
) -> np.ndarray[S, np.dtype[ND]]:
    """
    Binary search on a monotonic function that takes a NumPy array argument.

    The input to `fn` will have the shape determined by the `shape` argument, and must return output
    of the same shape. The arguments `target`, `low` and `high` must broadcastable to this shape.
    The result of this function will also have that shape.

    :param fn: Function to search.
    :param target: The value(s) to search for.
    :param low: If set defines the lowest possible input argument to search. If set, the function
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound. Must broadcast to `shape`.
    :param high: If set defines the highest possible input argument to search. If set, the function
        will *not* be called with this argument, unless `(low == high).any()`, though this value
        will be returned if no lower argument value produces the `target` value. If unset, an
        exponential search is performed for the lower bound. Must broadcast to `shape`.
    :param shape: The shape of the input to `fn` and the result of this function. If unset, it will
        be determined by broadcasting `target`, `low` and `high` together (if those are set).
    :param dtype: The `dtype` to call `fn` with, and the result of this function. If unset, it will
        be determined by the `dtype` of `low` and `high` (if those are set).
    :param side: If "left", returns the lowest argument value that produces a value greater than or
        equal to `target`. If "right" returns the lowest argument value that produces a value
        strictly greater than `target`.
    :param ordering: Whether the function outputs are "ascending" or "descending".
    """
    # pylint: disable=unnecessary-lambda-assignment

    target = np.asarray(target)

    if shape is None:
        known_arrays = [target]
        if low is not None:
            known_arrays.append(np.asarray(low))
        if high is not None:
            known_arrays.append(np.asarray(high))
        shape = np.broadcast_shapes(*[a.shape for a in known_arrays])

    if ordering == "ascending":
        if side == "left":
            pred = lambda x: target <= fn(x)
        elif side == "right":
            pred = lambda x: target < fn(x)
        else:
            assert_never(side)
    elif ordering == "descending":
        if side == "left":
            pred = lambda x: fn(x) <= target
        elif side == "right":
            pred = lambda x: fn(x) < target
        else:
            assert_never(side)
    else:
        assert_never(ordering)

    return search_numpy_pred(
        pred,
        low=low,
        high=high,
        shape=shape,
        dtype=dtype,
    )


def search_numpy_pred(
    pred: Callable[[np.ndarray[S, np.dtype[ND]]], np.ndarray[S, np.dtype[np.bool]]],
    *,
    low: npt.ArrayLike | None = None,
    high: npt.ArrayLike | None = None,
    shape: int | Iterable[int] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray[S, np.dtype[ND]]:
    """
    Binary search on a predicate that takes a NumPy "array" argument.

    Assume there exists some `i`, so that for all `j<i` `pred(j)` is `False` and for all `j>=i`
    `pred(j)` is `True`. This function uses binary search to find this `i`.

    The input to `pred` will have the shape determined by the `shape` argument, and must return
    output of the same shape. The arguments `low` and `high` must broadcastable to this shape. The
    result of this function will also have that shape.

    :param pred: Predicate to search.
    :param low: If set defines the lowest possible input argument to search. If set, the predicate
        *will* be called with this argument. If unset, an exponential search is performed for the
        lower bound. Must broadcast to `shape`.
    :param high: If set defines the highest possible input argument to search. If set, the predicate
        will *not* be called with this argument, unless `(low == high).any()`, though this value
        will be returned if no lower argument value produces the `target` value. If unset, an
        exponential search is performed for the lower bound. Must broadcast to `shape`.
    :param shape: The shape of the input to `pred` and the result of this function. If unset, it
        will be determined by broadcasting `target`, `low` and `high` together (if those are set).
    :param dtype: The `dtype` to call `pred` with, and the result of this function. If unset, it
        will be determined by the `dtype` of `low` and `high` (if those are set).
    """
    known_arrays = {}

    if dtype is not None:
        dtype = np.dtype(dtype)

    if low is not None:
        low = np.asarray(low, dtype=dtype)
        known_arrays["low"] = low

    if high is not None:
        high = np.asarray(high, dtype=dtype)
        known_arrays["high"] = high

    if shape is None:
        if not known_arrays:
            raise ValueError("Unable to infer shape. Please set the 'shape' parameter.")
        shape = np.broadcast_shapes(*[a.shape for a in known_arrays.values()])
    elif isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)

    if dtype is None:
        dtypes = {a.dtype for a in known_arrays.values()}
        if len(dtypes) != 1:
            raise ValueError("Unable to infer dtype. Please set the 'dtype' parameter.")
        (dtype,) = dtypes
    else:
        assert all(dtype == a.dtype for a in known_arrays.values()), (
            dtype,
            {n: a.dtype for n, a in known_arrays.items()},
        )

    ops = _get_operations(dtype)

    if low is not None and high is not None:
        assert (low <= high).all(), (low, high)

    result = np.zeros(shape, dtype)
    result_valid = np.full(shape, False, dtype=np.bool_)

    if low is None:
        low = np.full(shape, ops.min_value, dtype)
        low_valid = np.full(shape, False, dtype=np.bool_)
    else:
        low = np.broadcast_to(np.asarray(low, dtype=dtype), shape)
        low_valid = np.full(shape, True, dtype=np.bool_)
    p = pred(low)
    result = np.where(p, low, result)
    result_valid |= p
    low_valid |= p

    if high is None:
        high = np.full(shape, ops.max_value, dtype)
        high_valid = np.full(shape, False, dtype=np.bool_)
    else:
        high = np.broadcast_to(np.asarray(high, dtype=dtype), shape)
        high_valid = np.full(shape, True, dtype=np.bool_)
    p = pred(ops.prev(high))
    result = np.where(p, result, high)
    result_valid |= ~p
    high_valid |= ~p

    while True:
        assert (low <= high).all(), (low, high)

        with np.errstate(all="ignore"):
            mid = np.where(
                low_valid,
                np.where(high_valid, ops.suggest(low, high), ops.suggest_no_high(low)),
                np.where(high_valid, ops.suggest_no_low(high), ops.suggest_no_low_high),
            )
        if (result_valid | (low_valid & (mid == low))).all():
            break

        p = pred(mid)
        low = np.where(p, low, mid)
        low_valid |= ~p
        high = np.where(p, mid, high)
        high_valid |= p

    assert (result_valid | high_valid).all(), (result_valid, high_valid)
    return np.where(result_valid, result, high)
