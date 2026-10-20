"""Feature standardization shared by the CLI pipeline and the experiment scripts."""

import numpy as np


def standardize_splits(x_train, *others):
    """Impute NaNs with the train mean and z-score every split with train stats.

    Only the training split is used to fit the statistics, so validation and
    test stay untouched by model selection.  Constant columns keep ``std = 1``.

    Returns:
        list[np.ndarray]: ``[x_train, *others]`` as standardized float32 arrays.
    """
    xs = [np.asarray(x, dtype=np.float32) for x in (x_train, *others)]
    mean = np.nanmean(xs[0], axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    xs = [np.where(np.isnan(x), mean, x) for x in xs]
    std = xs[0].std(axis=0)
    std[std == 0] = 1.0
    return [((x - mean) / std).astype(np.float32) for x in xs]
