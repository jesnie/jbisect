import json
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from math import inf, nextafter
from pathlib import Path
from typing import Any, Callable, Generic, ParamSpec, TypeAlias, TypeVar

import pytest

from jbisect import (
    Ordering,
    Side,
    bisect_float_bool_fn,
    bisect_float_fn,
    bisect_int_bool_fn,
    bisect_int_fn,
    bisect_seq,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RECORD_FILE = DATA_DIR / "records.json"


P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")
N = TypeVar("N", int, float)


Json: TypeAlias = Any


SIDES: tuple[Side, ...] = ("left", "right")


class CallRecorder(Generic[T, R]):

    def __init__(self, fn: Callable[[T], R]) -> None:
        self._fn = fn
        self.calls: list[T] = []

    def reset(self) -> None:
        self.calls = []

    def __call__(self, p: T) -> R:
        self.calls.append(p)
        return self._fn(p)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def le(x: N) -> CallRecorder[N, bool]:
    return CallRecorder(lambda y: y <= x)


def lt(x: N) -> CallRecorder[N, bool]:
    return CallRecorder(lambda y: y < x)


def ge(x: N) -> CallRecorder[N, bool]:
    return CallRecorder(lambda y: y >= x)


def itv(frm: N, to: N) -> CallRecorder[N, N]:
    def f(x: N) -> N:
        if x < frm:
            sub = frm
        elif x <= to:
            sub = x
        else:
            sub = to
        return x - sub

    return CallRecorder(f)


def slf(_: N) -> CallRecorder[N, N]:
    return CallRecorder(lambda y: y)


def neg(_: N) -> CallRecorder[N, N]:
    return CallRecorder(lambda y: -y)


def float_prev(x: float) -> float:
    return nextafter(x, -inf)


def clamp(low: N | None, high: N | None, value: N, side: Side) -> N:
    if (low is not None) and value < low:
        return low
    if (high is not None) and value >= high:
        return high
    if side == "right":
        value = (value + 1) if isinstance(value, int) else nextafter(value, inf)
    return value


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

        assert self._expected_value == value, (self._expected_value, value)
        n_calls = max((cr.n_calls for cr in self._call_recorders), default=0)
        assert n_calls <= self._max_n_calls, (n_calls, self._max_n_calls)

        return {
            "value": value,
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
    expected_left: int,
    expected_right: int,
    max_n_calls: int,
) -> CaseIter:
    yield Case(
        "left_" + name,
        bisect_seq,
        seq,
        target,
        low=low,
        high=high,
        side="left",
        ordering=ordering,
        expected_value=expected_left,
        max_n_calls=max_n_calls,
    )
    yield Case(
        "right_" + name,
        bisect_seq,
        seq,
        target,
        low=low,
        high=high,
        side="right",
        ordering=ordering,
        expected_value=expected_right,
        max_n_calls=max_n_calls,
    )


def make_float_cases(
    name: str, low: float, high: float, value: float, max_n_calls: int
) -> CaseIter:
    for low_none in [True, False]:
        for high_none in [True, False]:
            low_ = None if low_none else low
            high_ = None if high_none else high
            name_ = f"{low_none}_{high_none}_{name}"
            yield Case(
                f"le_{name_}",
                bisect_float_bool_fn,
                le(value),
                low=low_,
                high=high_,
                ordering="ascending",
                expected_value=clamp(low_, high_, value, "right"),
                max_n_calls=max_n_calls,
            )
            yield Case(
                f"lt_{name_}",
                bisect_float_bool_fn,
                lt(value),
                low=low_,
                high=high_,
                ordering="ascending",
                expected_value=clamp(low_, high_, float_prev(value), "right"),
                max_n_calls=max_n_calls,
            )
            yield Case(
                f"ge_{name_}",
                bisect_float_bool_fn,
                ge(value),
                low=low_,
                high=high_,
                ordering="descending",
                expected_value=clamp(low_, high_, float_prev(value), "right"),
                max_n_calls=max_n_calls,
            )
            for side in SIDES:
                yield Case(
                    f"slf_{side}_{name_}",
                    bisect_float_fn,
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
                    f"neg_{side}_{name_}",
                    bisect_float_fn,
                    neg(value),
                    -value,
                    low=low_,
                    high=high_,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=max_n_calls,
                )


def make_float_itv_cases(
    name: str, low: float, high: float, left: float, right: float, max_n_calls: int
) -> CaseIter:
    for low_none in [True, False]:
        for high_none in [True, False]:
            for side, value in zip(SIDES, [left, right]):
                low_ = None if low_none else low
                high_ = None if high_none else high
                name_ = f"{side}_{low_none}_{high_none}_{name}"
                yield Case(
                    f"itv_{name_}",
                    bisect_float_fn,
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
                    f"neg_itv_{name_}",
                    bisect_float_fn,
                    neg(value),
                    -value,
                    low=low_,
                    high=high_,
                    side=side,
                    ordering="descending",
                    expected_value=clamp(low_, high_, value, side),
                    max_n_calls=max_n_calls,
                )


def make_int_cases(
    name: str,
    low: int,
    high: int,
    value: int,
    int_max_n_calls: int,
    float_max_n_calls: int,
) -> CaseIter:
    yield Case(
        "le_" + name,
        bisect_int_bool_fn,
        le(value),
        low=low,
        high=high,
        ordering="ascending",
        expected_value=clamp(low, high, value, "right"),
        max_n_calls=int_max_n_calls,
    )
    yield Case(
        "ge_" + name,
        bisect_int_bool_fn,
        ge(value),
        low=low,
        high=high,
        ordering="descending",
        expected_value=clamp(low, high, value - 1, "right"),
        max_n_calls=int_max_n_calls,
    )
    for side in SIDES:
        yield Case(
            "slf_" + side + "__" + name,
            bisect_int_fn,
            slf(value),
            value,
            low=low,
            high=high,
            side=side,
            ordering="ascending",
            expected_value=clamp(low, high, value, side),
            max_n_calls=int_max_n_calls,
        )
        yield Case(
            "neg_" + side + "__" + name,
            bisect_int_fn,
            neg(value),
            -value,
            low=low,
            high=high,
            side=side,
            ordering="descending",
            expected_value=clamp(low, high, value, side),
            max_n_calls=int_max_n_calls,
        )
    r = range(low, high)
    try:
        yield from make_seq_cases(
            name,
            r,
            value,
            low=0,
            high=len(r),
            ordering="ascending",
            expected_left=clamp(0, len(r), value - low, "left"),
            expected_right=clamp(0, len(r), value - low, "right"),
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
    for side, value in zip(SIDES, [left, right]):
        yield Case(
            "itv_" + side + "__" + name,
            bisect_int_fn,
            slf(value),
            value,
            low=low,
            high=high,
            side=side,
            ordering="ascending",
            expected_value=clamp(low, high, value, side),
            max_n_calls=int_max_n_calls,
        )
        yield Case(
            "neg_itv_" + side + "__" + name,
            bisect_int_fn,
            neg(value),
            -value,
            low=low,
            high=high,
            side=side,
            ordering="descending",
            expected_value=clamp(low, high, value, side),
            max_n_calls=int_max_n_calls,
        )
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
# ONLY = {bisect_int_bool_fn}
# ONLY = {bisect_int_bool_fn, bisect_int_fn, bisect_seq}
# ONLY = {bisect_float_bool_fn}
# ONLY = {bisect_float_bool_fn, bisect_float_fn}
ONLY = None


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
def test_bisect(case: AnyCase) -> None:
    result = case.test()
    records = read_records()
    assert records[case.__name__] == result


if __name__ == "__main__":
    write_records()