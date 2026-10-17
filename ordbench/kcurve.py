"""Personalisation from K calibration answers (the thesis's K-curve) on the ProVoice study, in the frozen-feature setting.

For a held-out test driver the other drivers split into 9 training drivers (all their answers) and 2 validation drivers.
The test driver and the two validation drivers are in the same position: only K of their answers (the calibration
answers) are in the training set and their remaining answers are the evaluation items, so checkpoint and hyper-parameter
selection on the validation drivers rehearses exactly the situation of the test driver. A driver code (one-hot) is a
feature, and the L2 penalty shrinks its weight towards the population model; K = 0 means no calibration answer, and then
the three drivers carry a zero code, which is the population model. Calibration answers are nested prefixes of one random
order per driver, so the K-curve is paired across K. The other prompt of a calibration answer's 20 s window shares its
frames and is dropped from evaluation, so no frame of a calibration window is tested.
"""
import numpy as np

KS = (0, 1, 2, 5, 10, 20, 40)
ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def feature_matrix(X, names, drivers, kind):
    """Standardisable features and the raw driver code. kind: 'task' (the one-hot task code) or 'all' (every feature)."""
    names = [str(n) for n in names]
    cols = [i for i, n in enumerate(names) if n.startswith("task_")] if kind == "task" else list(range(len(names)))
    users = sorted(set(drivers))
    code = np.stack([(drivers == d).astype(np.float32) for d in users], 1)
    return np.asarray(X, np.float32)[:, cols], code, users


def kshot_split(drivers, windows, test_driver, seed, k, n_val=2):
    """Row indices: training (9 full drivers plus the K calibration answers of each personalised driver), validation, test,
    and the calibration rows of the personalised drivers {driver: rows}."""
    users = sorted(set(drivers))
    rng = np.random.default_rng([seed, users.index(test_driver)])
    others = [u for u in users if u != test_driver]
    val = [others[i] for i in rng.choice(len(others), n_val, replace=False)]
    personal = [test_driver] + val
    order = {d: rng.permutation(np.flatnonzero(drivers == d)) for d in personal}      # same for every K: nested prefixes
    cal = {d: order[d][:k] for d in personal}
    full = np.flatnonzero(~np.isin(drivers, personal))
    train = np.concatenate([full] + [cal[d] for d in personal])

    def evaluation_rows(d):
        rows = np.flatnonzero(drivers == d)
        twins = np.isin(windows[rows], windows[cal[d]]) if k else np.zeros(len(rows), bool)
        return rows[~np.isin(rows, cal[d]) & ~twins]
    return train, np.concatenate([evaluation_rows(d) for d in val]), evaluation_rows(test_driver), cal, val, personal


def driver_code(code, drivers, personal, k):
    """Driver code with the personalised drivers' codes removed when they have no calibration answer."""
    c = code.copy()
    if k == 0:
        c[np.isin(drivers, personal)] = 0
    return c


def personal_prior(soft, drivers, rows, cal, pop, alpha):
    """Row-wise log-PMF of the personal prior: the driver's calibration soft labels smoothed towards the population
    marginal, (sum of calibration labels + alpha * population) / (K + alpha); (1, len(rows), K) for `evaluate`."""
    out = np.empty((len(rows), soft.shape[1]))
    for d in set(drivers[rows]):
        n = len(cal[d]) if d in cal else 0
        counts = soft[cal[d]].sum(0) if n else np.zeros(soft.shape[1])
        out[drivers[rows] == d] = (counts + alpha * pop) / (n + alpha)
    return np.log(np.clip(out, 1e-12, None))[None]


def collapse_sets(mask, rule, seed=0):
    """Replace every accepted set by one of its levels (a control that removes the set information from training):
    'lower' / 'upper' median of the set, or a uniformly random member."""
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    rng = np.random.default_rng(seed)
    for i, row in enumerate(mask):
        levels = np.flatnonzero(row)
        if rule == "lower":
            pick = levels[(len(levels) - 1) // 2]
        elif rule == "upper":
            pick = levels[len(levels) // 2]
        elif rule == "random":
            pick = rng.choice(levels)
        else:
            raise ValueError(rule)
        out[i, pick] = True
    return out
