import numpy as np

from jbisect.numpy import search_numpy_array, search_numpy_fn, search_numpy_pred


def test_search_numpy_pred__base() -> None:
    assert search_numpy_pred(lambda i: i >= 4, shape=[], dtype=np.int64) == np.array(
        4, dtype=np.int64
    )


def test_search_numpy_pred__shape() -> None:
    assert (
        search_numpy_pred(lambda i: i >= 4, shape=[3], dtype=np.int64)
        == np.array([4, 4, 4], dtype=np.int64)
    ).all()
    assert (
        search_numpy_pred(lambda i: i >= 4, shape=[2, 3], dtype=np.int64)
        == np.array([[4, 4, 4], [4, 4, 4]], dtype=np.int64)
    ).all()


def test_search_numpy_pred__low() -> None:
    assert search_numpy_pred(lambda i: i >= 4, low=4) == np.array(4, dtype=np.int64)
    assert search_numpy_pred(lambda i: i >= 4, low=5) == np.array(5, dtype=np.int64)
    assert (
        search_numpy_pred(lambda i: i >= 4, low=[3, 4, 5]) == np.array([4, 4, 5], dtype=np.int64)
    ).all()


def test_search_numpy_pred__high() -> None:
    assert search_numpy_pred(lambda i: i >= 4, high=4) == np.array(4, dtype=np.int64)
    assert search_numpy_pred(lambda i: i >= 4, high=3) == np.array(3, dtype=np.int64)
    assert (
        search_numpy_pred(lambda i: i >= 4, high=[3, 4, 5]) == np.array([3, 4, 4], dtype=np.int64)
    ).all()


def test_search_numpy_pred__int8() -> None:
    t = np.array(range(-128, 128), np.int8)
    assert (search_numpy_pred(lambda i: i >= t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_pred__uint8() -> None:
    t = np.array(range(256), dtype=np.uint8)
    assert (search_numpy_pred(lambda i: i >= t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_pred__float32() -> None:
    t = np.array([-1.0e30, -1.0, 0.0, 1.0e-30, 1.0, 1.0e30], dtype=np.float32)
    assert (search_numpy_pred(lambda i: i >= t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_fn__base() -> None:
    assert search_numpy_fn(lambda i: i, 4, shape=[], dtype=np.int64) == np.array(4, dtype=np.int64)


def test_search_numpy_fn__shape() -> None:
    assert (
        search_numpy_fn(lambda i: i, 4, shape=[3], dtype=np.int64)
        == np.array([4, 4, 4], dtype=np.int64)
    ).all()
    assert (
        search_numpy_fn(lambda i: i, 4, shape=[2, 3], dtype=np.int64)
        == np.array([[4, 4, 4], [4, 4, 4]], dtype=np.int64)
    ).all()
    assert (
        search_numpy_fn(
            lambda i: i * i,
            [[[4, 6, 8, 10]]],
            low=[[[1], [2], [3]]],
            high=[[[3]], [[4]]],
        )
        == np.array(
            [
                [
                    [2, 3, 3, 3],
                    [2, 3, 3, 3],
                    [3, 3, 3, 3],
                ],
                [
                    [2, 3, 3, 4],
                    [2, 3, 3, 4],
                    [3, 3, 3, 4],
                ],
            ],
            dtype=np.int_,
        )
    ).all()


def test_search_numpy_fn__low() -> None:
    assert search_numpy_fn(lambda i: i, 4, low=4) == np.array(4, dtype=np.int64)
    assert search_numpy_fn(lambda i: i, 4, low=5) == np.array(5, dtype=np.int64)
    assert (
        search_numpy_fn(lambda i: i, 4, low=[3, 4, 5]) == np.array([4, 4, 5], dtype=np.int64)
    ).all()


def test_search_numpy_fn__high() -> None:
    assert search_numpy_fn(lambda i: i, 4, high=4) == np.array(4, dtype=np.int64)
    assert search_numpy_fn(lambda i: i, 4, high=3) == np.array(3, dtype=np.int64)
    assert (
        search_numpy_fn(lambda i: i, 4, high=[3, 4, 5]) == np.array([3, 4, 4], dtype=np.int64)
    ).all()


def test_search_numpy_fn__side() -> None:
    assert (
        search_numpy_fn(
            lambda i: i * i,
            16,
            low=0,
            high=1000,
            shape=(),
            dtype=np.int64,
            side="right",
        )
        == 5
    )


def test_search_numpy_fn__ordering() -> None:
    assert (
        search_numpy_fn(
            lambda i: -i * i,
            -16,
            low=0,
            high=1000,
            shape=(),
            dtype=np.int64,
            ordering="descending",
        )
        == 4
    )


def test_search_numpy_fn__int8() -> None:
    t = np.array(range(-128, 128), np.int8)
    assert (search_numpy_fn(lambda i: i, t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_fn__uint8() -> None:
    t = np.array(range(256), dtype=np.uint8)
    assert (search_numpy_fn(lambda i: i, t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_fn__float32() -> None:
    t = np.array([-1.0e30, -1.0, 0.0, 1.0e-30, 1.0, 1.0e30], dtype=np.float32)
    assert (search_numpy_fn(lambda i: i, t, shape=t.shape, dtype=t.dtype) == t).all()


def test_search_numpy_array() -> None:
    assert search_numpy_array([], 2) == 0
    assert search_numpy_array([2], 2) == 0
    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2) == 3

    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2, low=3) == 3
    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2, low=4) == 4

    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2, high=3) == 3
    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2, high=2) == 2

    assert search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2, side="right") == 6

    assert search_numpy_array([5, 5, 3, 2, 2, 2, 1, 1, 0], 2, ordering="descending") == 3

    # 2D:
    assert (
        search_numpy_array(
            [
                [111, 112, 113, 114],
                [121, 122, 123, 124],
                [131, 132, 133, 134],
            ],
            [100, 110, 120, 130],
            axis=0,
        )
        == np.array([0, 0, 1, 2], dtype=np.uintp)
    ).all()
    assert (
        search_numpy_array(
            [
                [111, 112, 113, 114],
                [121, 122, 123, 124],
                [131, 132, 133, 134],
            ],
            [112, 123, 134],
            axis=1,
        )
        == np.array([1, 2, 3], dtype=np.uintp)
    ).all()

    # 3D:
    assert (
        search_numpy_array(
            [
                [
                    [111, 112, 113, 114],
                    [121, 122, 123, 124],
                    [131, 132, 133, 134],
                ],
                [
                    [211, 212, 213, 214],
                    [221, 222, 223, 224],
                    [231, 232, 233, 234],
                ],
            ],
            200,
            axis=0,
        )
        == np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.uintp,
        )
    ).all()
    assert (
        search_numpy_array(
            [
                [
                    [111, 112, 113, 114],
                    [121, 122, 123, 124],
                    [131, 132, 133, 134],
                ],
                [
                    [211, 212, 213, 214],
                    [221, 222, 223, 224],
                    [231, 232, 233, 234],
                ],
            ],
            [[122], [223]],
            axis=1,
        )
        == np.array(
            [
                [2, 1, 1, 1],
                [2, 2, 1, 1],
            ],
            dtype=np.uintp,
        )
    ).all()
    assert (
        search_numpy_array(
            [
                [
                    [111, 112, 113, 114],
                    [121, 122, 123, 124],
                    [131, 132, 133, 134],
                ],
                [
                    [211, 212, 213, 214],
                    [221, 222, 223, 224],
                    [231, 232, 233, 234],
                ],
            ],
            [
                [111, 122, 133],
                [214, 222, 233],
            ],
            axis=2,
        )
        == np.array(
            [
                [0, 1, 2],
                [3, 1, 2],
            ],
            dtype=np.uintp,
        )
    ).all()
