"""Feature standardization shared by the CLI pipeline and the experiment scripts."""

import numpy as np


def standardize_splits(x_train, *others, inplace=False):
    """Impute NaNs with the train mean and z-score every split with train stats.

    Only the training split is used to fit the statistics, so validation and
    test stay untouched by model selection.  Constant columns keep ``std = 1``.

    With ``inplace=True`` the writable input arrays are standardized in place,
    avoiding the two full intermediate copies (which dominate memory on large
    text datasets such as RCV1).  Non-writable arrays are copied first.

    Returns:
        list[np.ndarray]: ``[x_train, *others]`` as standardized float32 arrays.
    """
    xs = [np.asarray(x, dtype=np.float32) for x in (x_train, *others)]
    mean = np.nanmean(xs[0], axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    if inplace:
        # std of the train split after mean-imputation, without materializing
        # the imputed copy: (x - mean) contributes zero for NaN entries.
        dev = np.where(np.isnan(xs[0]), 0.0, xs[0] - mean)
        std = np.sqrt((dev * dev).mean(axis=0))
        std[std == 0] = 1.0
        out = []
        for x in xs:
            if not x.flags.writeable:
                x = x.copy()
            x -= mean
            np.divide(x, std, out=x)
            np.nan_to_num(x, copy=False, nan=0.0, posinf=np.inf, neginf=-np.inf)
            out.append(x)
        return out
    xs = [np.where(np.isnan(x), mean, x) for x in xs]
    std = xs[0].std(axis=0)
    std[std == 0] = 1.0
    return [((x - mean) / std).astype(np.float32) for x in xs]
