"""The ProVoice builder must put every frame in the right 20 s window and turn the answers into accepted-level sets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import provoice as P  # noqa: E402


def make_frames(times, session="s1"):
    n = len(times)
    f = pd.DataFrame({c: np.zeros(n) for c in P.FRAME_COLUMNS if c not in ("timestamp", "session_id", "emotion",
                                                                          "traffic_light_state")})
    f["session_id"] = session
    f["emotion"] = "neutral"
    f["traffic_light_state"] = "Green"
    f["t"] = np.asarray(times, float)
    return f


def make_labels():
    base = dict(session_id="s1", participantid="007", window_start_timestamp="2026-08-07T10:00:00.000000",
                window_end_timestamp="2026-08-07T10:00:20.000000")
    second = dict(base, window_start_timestamp="2026-08-07T10:00:30.000000", window_end_timestamp="2026-08-07T10:00:50.000000")
    return pd.DataFrame([dict(base, window_idx=1, prompt_in_window=1, functionname="Change song", user_selected_loa="2"),
                         dict(base, window_idx=1, prompt_in_window=2, functionname="Provide weather update", user_selected_loa="1;2;3"),
                         dict(second, window_idx=2, prompt_in_window=1, functionname="Respond to a phone call", user_selected_loa="0;3")])


def test_seconds_of_day_reads_frame_stamps_and_iso_times():
    s = P.seconds_of_day(["17:04:49.932", "2026-08-07T17:04:59.508311"])
    assert s == pytest.approx([17 * 3600 + 4 * 60 + 49.932, 17 * 3600 + 4 * 60 + 59.508311])


def test_frames_are_assigned_to_windows_with_inclusive_bounds_and_gaps_dropped():
    t0 = 10 * 3600
    times = [t0 - 1, t0, t0 + 10, t0 + 20, t0 + 25, t0 + 30, t0 + 40, t0 + 51]
    f = make_frames(times)
    f["speed_kmh"] = [100, 1, 2, 3, 100, 4, 6, 100]       # frames outside both windows carry 100
    X, mask, ids, group, names = P.assemble(make_labels(), f)
    col = names.index("speed_kmh_mean")
    assert X[0, col] == pytest.approx(2.0) and X[1, col] == pytest.approx(2.0)     # window 1: 1, 2, 3
    assert X[2, col] == pytest.approx(5.0)                                          # window 2: 4, 6
    assert list(ids) == ["s1|win001p1", "s1|win001p2", "s1|win002p1"]
    assert set(group) == {"007"}


def test_answers_become_accepted_sets_and_tasks_are_one_hot():
    t0 = 10 * 3600
    X, mask, ids, group, names = P.assemble(make_labels(), make_frames([t0 + 5, t0 + 35]))
    assert mask.tolist() == [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 0]]
    task = X[:, [names.index(f"task_{t}") for t in P.TASKS]]
    assert task.sum(1).tolist() == [1, 1, 1]
    assert task[0, P.TASKS.index("Change song")] == 1 and task[2, P.TASKS.index("Respond to a phone call")] == 1


def test_missing_values_are_imputed_and_windows_without_frames_dropped():
    t0 = 10 * 3600
    f = make_frames([t0 + 5, t0 + 6, t0 + 35])
    f["heart_rate"] = [np.nan, np.nan, 70.0]
    f["lead_distance_m"] = [-1.0, -1.0, 12.0]
    X, mask, ids, group, names = P.assemble(make_labels(), f)
    assert np.isfinite(X).all()
    assert X[0, names.index("hr_available")] == 0 and X[2, names.index("hr_available")] == 1
    X2, mask2, ids2, _, _ = P.assemble(make_labels(), make_frames([t0 + 5]))          # nothing inside window 2
    assert len(X2) == 2 and list(ids2) == ["s1|win001p1", "s1|win001p2"]


def _fake_processed(tmp_path, monkeypatch):
    from ordbench import native, paths
    monkeypatch.setattr(paths, "PROCESSED", tmp_path)
    monkeypatch.setattr(native, "PROCESSED", tmp_path)
    g = np.random.default_rng(0)
    group = np.repeat(["001", "002", "003"], 8)
    state = g.normal(size=(24, 3)) + np.repeat([0.0, 5.0, -3.0], 8)[:, None]          # a different offset per driver
    task = np.eye(5)[g.integers(0, 5, 24)]
    ids = np.array([f"s{d}|win{w // 2:03d}p{w % 2 + 1}" for d, w in zip(group, range(24))])
    mask = np.zeros((24, 5), bool)
    mask[np.arange(24), g.integers(0, 5, 24)] = True
    names = ["a", "b", "c"] + [f"task_{t}" for t in P.TASKS]
    native.save_native("provoice", np.concatenate([state, task], 1), mask, 5, ids, group, extra=dict(feature_names=np.array(names)))
    return np.concatenate([state, task], 1), group, ids


def test_driver_centred_variant_removes_between_driver_offsets_only(tmp_path, monkeypatch):
    X, group, ids = _fake_processed(tmp_path, monkeypatch)
    P.build_provoice_centred()
    Z = np.load(tmp_path / "provoice_centred.npz", allow_pickle=True)["X"]
    for d in np.unique(group):
        assert np.abs(Z[group == d, :3].mean(0)).max() < 1e-5
    assert np.array_equal(Z[:, 3:], X[:, 3:].astype(np.float32))


def test_known_driver_variant_adds_driver_codes_and_splits_by_window(tmp_path, monkeypatch):
    X, group, ids = _fake_processed(tmp_path, monkeypatch)
    P.build_provoice_known()
    z = np.load(tmp_path / "provoice_known.npz", allow_pickle=True)
    assert z["X"].shape[1] == X.shape[1] + 3 and (z["X"][:, -3:].sum(1) == 1).all()
    assert len(set(z["group"])) == 12 and all(g.count("|win") == 1 and "p" not in g.rsplit("win", 1)[1] for g in z["group"])
