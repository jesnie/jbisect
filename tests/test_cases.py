import json
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from math import inf, nextafter
from pathlib import Path
from typing import Any, Callable, Generic, ParamSpec, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytest

from jbisect import (
    Ordering,
    Side,
    search_float_fn,
    search_float_pred,
    search_int_fn,
    search_int_pred,
    search_seq,
)
from jbisect.numpy import search_numpy_array, search_numpy_fn, search_numpy_pred

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RECORD_FILE = DATA_DIR / "records.json"


P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")
N = TypeVar("N", int, float)


Json: TypeAlias = Any


class CallRecorder(Generic[T, R]):

    def __init__(self, fn: Callable[[T], R], name: str) -> None:
        self._fn = fn
        self._name = name
        self.calls: list[T] = []

    def reset(self) -> None:
        self.calls = []

    def __call__(self, p: T) -> R:
        self.calls.append(p)
        return self._fn(p)

    def __repr__(self) -> str:
        return self._name

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def gt(x: N) -> CallRecorder[N, bool]:
    return CallRecorder(lambda y: y > x, f"gt({x})")


def le(x: N) -> CallRecorder[N, bool]:
    return CallRecorder(lambda y: y <= x, f"le({x})")


def itv(frm: N, to: N) -> CallRecorder[N, N]:
    def f(x: N) -> N:
        if x < frm:
            sub = frm
        elif x <= to:
            sub = x
        else:
            sub = to
        return x - sub

    return CallRecorder(f, f"itv({frm}, {to})")


def slf(x: N) -> CallRecorder[N, N]:
    return CallRecorder(lambda y: y, f"slf({x})")


def neg(x: N) -> CallRecorder[N, N]:

    def _neg(y: N) -> N:
        if y != 0 and -y == y:
            # Handle signed-int underflows:
            return -(y + 1)
        return -y

    return CallRecorder(_neg, f"neg({x})")


def clamp(low: N | None, high: N | None, value: N, side: Side) -> N:
    if (low is not None) and value < low:
        return low
    if (high is not None) and value >= high:
        return high
    if side == "right":
        value = (value + 1) if isinstance(value, int) else nextafter(value, inf)
    return value


def iter_limits(low: N, high: N, name: str) -> Iterator[tuple[N | None, N | None, str]]:
    yield low, high, name
    yield None, high, "nolow_" + name
    yield low, None, "nohigh_" + name
    yield None, None, "nolow_nohigh_" + name


def iter_sides(left: N, right: N, name: str) -> Iterator[tuple[Side, N, str]]:
    yield "left", left, "left_" + name
    yield "right", right, "right_" + name


def raise_on_overflow(*args: Any, dtype: npt.DTypeLike) -> None:
    values = [a for a in args if a is not None]
    if values:
        np.array(values, dtype=dtype)


class Case(Generic[P, R]):

    def __init__(
        self,
        name: str,
        target: Callable[P, R],
        *args: P.args,
        expected_value: R,
        max_n_calls: int,
        **kwargs: P.kwargs,
    ) -> None:
        self.name = f"{target.__name__}__{name}"
        self._target = target
        self._expected_value = expected_value
        self._max_n_calls = max_n_calls
        self._args = args
        self._kwargs = kwargs
        self._call_recorders = [a for a in args if isinstance(a, CallRecorder)] + [
            a for a in kwargs.values() if isinstance(a, CallRecorder)
        ]

    @property
    def __name__(self) -> str:
        return self.name

    @property
    def target(self) -> Callable[P, R]:
        return self._target

    def test(self) -> Json:
        for cr in self._call_recorders:
            cr.reset()

        value = self._target(*self._args, **self._kwargs)
        convert = getattr(value, "item", lambda: value)
        value = convert()

        assert self._expected_value == value, (self._expected_value, value)
        n_calls = max((cr.n_calls for cr in self._call_recorders), default=0)
        assert n_calls <= self._max_n_calls, (n_calls, self._max_n_calls)

        arg_exprs = [f"{a!r}" for a in self._args] + [f"{n}={a!r}" for n, a in self._kwargs.items()]
        expr = f"{self._target.__name__}({', '.join(arg_exprs)})"

        return {
            "expr": expr,
            "result": value,
            "n_calls": n_calls,
        }


AnyCase: TypeAlias = Case[Any, Any]
CaseIter: TypeAlias = Iterator[AnyCase]


def make_seq_cases(
    name: str,
    seq: Sequence[T],
    target: T,
    low: int,
    high: int,
    ordering: Ordering,
    left: int,
    right: int,
    max_n_calls: int,
) -> CaseIter:
    assert 0 <= low <= high <= len(seq)
    for low_, high_, name_ in iter_limits(low, high, name):
        for side, value, name__ in iter_sides(left, right, name_):
            yield Case(
                name__,
                search_seq,
                seq,
                target,
                low=low_,
                high=high_,
                side=side,
                ordering=ordering,
                expected_value=value,
                max_n_calls=max_n_calls,
            )
            try:
                if isinstance(seq, str):
                    seq_: Sequence[Any] = [ord(c) for c in cast(str, seq)]
                    target_: Any = ord(cast(str, target))
                else:
                    seq_ = seq
                    target_ = target
                yield Case(
                    name__,
                    search_numpy_array,
                    seq_,
                    target_,
                    low=low_,
                    high=high_,
                    side=side,
                    ordering=ordering,
                    expected_value=value,
                    max_n_calls=max_n_calls,
                )
            except OverflowError:
                pass


def make_float_cases(
    name: str, low: float, high: float, value: float, max_n_calls: int
) -> CaseIter:
    for low_, high_, name_ in iter_limits(low, high, name):
        yield Case(
            f"gt_{name_}",
            search_float_pred,
            gt(value),
            low=low_,
            high=high_,
            expected_value=clamp(low_, high_, value, "right"),
            max_n_calls=max_n_calls,
        )
        for side, value_, name__ in iter_sides(value, value, name_):
            yield Case(
                f"slf_{name__}",
                search_float_fn,
                slf(value_),
                value_,
                low=low_,
                high=high_,
                side=side,
                ordering="ascending",
                expected_value=clamp(low_, high_, value_, side),
                max_n_calls=max_n_calls,
            )
            yield Case(
                f"neg_{name__}",
                search_float_fn,
                neg(value_),
                -value_,
                low=low_,
                high=high_,
                side=side,
                ordering="descending",
                expected_value=clamp(low_, high_, value_, side),
                max_n_calls=max_n_calls,
            )
        try:
            raise_on_overflow(low_, high_, value, dtype=np.float64)
            yield Case(
                f"float64_gt_{name_}",
                search_numpy_pred,
                gt(value),
                low=low_,
                high=high_,
                shape=(),
                dtype=np.float64,
                expected_value=clamp(low_, high_, value, "right"),
                max_n_calls=max_n_calls,
            )
            for side, value_, name__ in iter_sides(value, value, name_):
                yield Case(
                    f"float64_slf_{name__}",
                    search_numpy_fn,
                    slf(value_),
                    value_,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.float64,
                    side=side,
                    ordering="ascending",
                    expected_value=clamp(low_, high_, value_, side),
                    max_n_calls=max_n_calls,
                )
                yield Case(
                    f"float64_neg_{name__}",
                    search_numpy_fn,
                    neg(value_),
                    -value_,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.float64,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value_, side),
                    max_n_calls=max_n_calls,
                )
        except OverflowError:
            pass


def make_float_itv_cases(
    name: str, low: float, high: float, left: float, right: float, max_n_calls: int
) -> CaseIter:
    for low_, high_, name_ in iter_limits(low, high, name):
        for side, value, name__ in iter_sides(left, right, name_):
            yield Case(
                f"itv_{name__}",
                search_float_fn,
                slf(value),
                value,
                low=low_,
                high=high_,
                side=side,
                ordering="ascending",
                expected_value=clamp(low_, high_, value, side),
                max_n_calls=max_n_calls,
            )
            yield Case(
                f"neg_itv_{name__}",
                search_float_fn,
                neg(value),
                -value,
                low=low_,
                high=high_,
                side=side,
                ordering="descending",
                expected_value=clamp(low_, high_, value, side),
                max_n_calls=max_n_calls,
            )
            try:
                raise_on_overflow(low_, high_, value, dtype=np.float64)
                yield Case(
                    f"float64_itv_{name__}",
                    search_numpy_fn,
                    slf(value),
                    value,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.float64,
                    side=side,
                    ordering="ascending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=max_n_calls,
                )
                yield Case(
                    f"float64_neg_itv_{name__}",
                    search_numpy_fn,
                    neg(value),
                    -value,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.float64,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=max_n_calls,
                )
            except OverflowError:
                pass


def make_int_cases(
    name: str,
    low: int,
    high: int,
    value: int,
    int_max_n_calls: int,
    float_max_n_calls: int,
) -> CaseIter:
    for low_, high_, name_ in iter_limits(low, high, name):
        yield Case(
            "gt_" + name_,
            search_int_pred,
            gt(value),
            low=low_,
            high=high_,
            expected_value=clamp(low_, high_, value, "right"),
            max_n_calls=int_max_n_calls,
        )
        for side, value_, name__ in iter_sides(value, value, name_):
            yield Case(
                "slf_" + name__,
                search_int_fn,
                slf(value_),
                value_,
                low=low_,
                high=high_,
                side=side,
                ordering="ascending",
                expected_value=clamp(low_, high_, value_, side),
                max_n_calls=int_max_n_calls,
            )
            yield Case(
                "neg_" + name__,
                search_int_fn,
                neg(value_),
                -value_,
                low=low_,
                high=high_,
                side=side,
                ordering="descending",
                expected_value=clamp(low_, high_, value_, side),
                max_n_calls=int_max_n_calls,
            )
        try:
            raise_on_overflow(low_, high_, value, dtype=np.int64)
            yield Case(
                "int64_gt_" + name_,
                search_numpy_pred,
                gt(value),
                low=low_,
                high=high_,
                shape=(),
                dtype=np.int64,
                expected_value=clamp(low_, high_, value, "right"),
                max_n_calls=int_max_n_calls,
            )
            for side, value_, name__ in iter_sides(value, value, name_):
                yield Case(
                    "int64_slf_" + name__,
                    search_numpy_fn,
                    slf(value_),
                    value_,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.int64,
                    side=side,
                    ordering="ascending",
                    expected_value=clamp(low_, high_, value_, side),
                    max_n_calls=int_max_n_calls,
                )
                yield Case(
                    "int64_neg_" + name__,
                    search_numpy_fn,
                    neg(value_),
                    -value_,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.int64,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value_, side),
                    max_n_calls=int_max_n_calls,
                )
        except OverflowError:
            pass

    try:
        r = range(low, high)
        low_ = 0
        high_ = len(r)
        yield from make_seq_cases(
            name,
            r,
            value,
            low=low_,
            high=high_,
            ordering="ascending",
            left=clamp(low_, high_, value - low, "left"),
            right=clamp(low_, high_, value - low, "right"),
            max_n_calls=int_max_n_calls,
        )
    except OverflowError:
        pass

    yield from make_float_cases(name, float(low), float(high), float(value), float_max_n_calls)


def make_int_itv_cases(
    name: str,
    low: int,
    high: int,
    left: int,
    right: int,
    int_max_n_calls: int,
    float_max_n_calls: int,
) -> CaseIter:
    for low_, high_, name_ in iter_limits(low, high, name):
        for side, value, name__ in iter_sides(left, right, name_):
            yield Case(
                "itv_" + name__,
                search_int_fn,
                slf(value),
                value,
                low=low_,
                high=high_,
                side=side,
                ordering="ascending",
                expected_value=clamp(low_, high_, value, side),
                max_n_calls=int_max_n_calls,
            )
            yield Case(
                "neg_itv_" + name__,
                search_int_fn,
                neg(value),
                -value,
                low=low_,
                high=high_,
                side=side,
                ordering="descending",
                expected_value=clamp(low_, high_, value, side),
                max_n_calls=int_max_n_calls,
            )

            try:
                raise_on_overflow(low_, high_, value, dtype=np.int64)
                yield Case(
                    "int64_itv_" + name__,
                    search_numpy_fn,
                    slf(value),
                    value,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.int64,
                    side=side,
                    ordering="ascending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=int_max_n_calls,
                )
                yield Case(
                    "int64_neg_itv_" + name__,
                    search_numpy_fn,
                    neg(value),
                    -value,
                    low=low_,
                    high=high_,
                    shape=(),
                    dtype=np.int64,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=int_max_n_calls,
                )
            except OverflowError:
                pass

    yield from make_float_itv_cases(
        name, float(low), float(high), float(left), float(right), float_max_n_calls
    )


def make_cases() -> tuple[AnyCase, ...]:
    result: list[AnyCase] = []

    for name, low, high, values in [
        ("empty_positive", 100, 100, [99, 100, 101]),
        ("empty_zero", 0, 0, [-1, 0, 1]),
        ("empty_negative", -100, -100, [-101, -100, -99]),
        ("singleton", 0, 1, [-1, 0, 1, 2]),
        ("small", 0, 3, [-1, 0, 1, 2, 3, 4]),
        ("positive", 100, 200, [99, 100, 101, 150, 199, 200, 201]),
        ("nonnegative", 0, 100, [-1, 0, 1, 50, 99, 100, 101]),
        ("stride", -50, 50, [-51, -50, -49, 0, 49, 50, 51]),
        ("nonpositive", -100, 0, [-101, -100, -99, -50, -1, 0, 1]),
        ("negative", -200, -100, [-201, -200, -199, -150, -101, -100, -99]),
    ]:
        for value in values:
            result.extend(
                make_int_cases(
                    f"{name}_{value}", low, high, value, int_max_n_calls=15, float_max_n_calls=70
                )
            )

    for name, flow, fhigh, fvalues in [("powers", 0.0, 1.0, [1 / (10**i) for i in range(5)])]:
        for fvalue in fvalues:
            result.extend(make_float_cases(f"{name}_{fvalue}", flow, fhigh, fvalue, max_n_calls=70))

    result.extend(
        make_int_cases(
            "huge_range",
            -(10**100),
            10**100,
            31337,
            int_max_n_calls=500,
            float_max_n_calls=70,
        )
    )
    result.extend(make_float_cases("huge_scale", -1e-100, 1e100, 2e-52, max_n_calls=70))

    for name, low, high, itv_values in [
        ("empty_interval", 0, 0, [(-2, -1), (-1, 1), (0, 0), (1, 2)]),
        ("singleton_interval", 0, 1, [(-2, -1), (-1, 2), (0, 0), (0, 1), (1, 1), (1, 2)]),
        (
            "interval",
            -100,
            100,
            [(-110, -100), (-110, -90), (-50, -40), (-10, 10), (40, 50), (90, 110), (100, 110)],
        ),
    ]:
        for left, right in itv_values:
            result.extend(
                make_int_itv_cases(
                    f"{name}_{left}_{right}",
                    low,
                    high,
                    left,
                    right,
                    int_max_n_calls=15,
                    float_max_n_calls=70,
                )
            )

    for name, seq, targets in [
        ("empty_sequence", "", [("0", 0, 0)]),
        ("singleton_sequence", "5", [("4", 0, 0), ("5", 0, 1), ("6", 1, 1)]),
        (
            "sequence",
            "1223335555",
            [
                ("0", 0, 0),
                ("1", 0, 1),
                ("2", 1, 3),
                ("3", 3, 6),
                ("4", 6, 6),
                ("5", 6, 10),
                ("6", 10, 10),
            ],
        ),
    ]:
        for target, expected_left, expected_right in targets:
            result.extend(
                make_seq_cases(
                    f"{name}_{target}",
                    seq,
                    target,
                    0,
                    len(seq),
                    "ascending",
                    expected_left,
                    expected_right,
                    10,
                )
            )

    return tuple(result)


CASES = make_cases()
ONLY: set[Callable[..., Any]] | None
ONLY = None
# ONLY = {search_int_pred}
# ONLY = {search_int_pred, search_int_fn, search_seq}
# ONLY = {search_float_pred}
# ONLY = {search_float_pred, search_float_fn}
# ONLY = {search_numpy_pred}


def filter_cases() -> Sequence[AnyCase]:
    if ONLY is None:
        return CASES
    return [case for case in CASES if case.target in ONLY]


@cache
def read_records() -> Mapping[str, Json]:
    with open(RECORD_FILE, "rt", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def write_records() -> None:
    RECORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for case in filter_cases():
        print("--------------------------------------------------")
        print(case.__name__)
        records[case.__name__] = case.test()
    with open(RECORD_FILE, "wt", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


@pytest.mark.parametrize("case", filter_cases())
def test_case(case: AnyCase) -> None:
    result = case.test()
    records = read_records()
    assert records[case.__name__] == result


if __name__ == "__main__":
    write_records()
