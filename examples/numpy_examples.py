import numpy as np

from jbisect.numpy import search_numpy_array, search_numpy_fn, search_numpy_pred

print(search_numpy_array([0, 1, 1, 2, 2, 2, 3, 5, 5], 2))

print(
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
)

print(search_numpy_fn(lambda i: i * i, 16, low=0, high=1000, shape=(), dtype=np.int64))

print(search_numpy_pred(lambda i: i >= 4, shape=[], dtype=np.int64))
