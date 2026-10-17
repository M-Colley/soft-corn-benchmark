"""ProVoice study 1 (in-vehicle assistant, 12 drivers x 2 sessions): the benchmark's natively set-valued dataset.

Every 20 s the driver was asked about two in-vehicle tasks and marked, for each, the levels of autonomy (LoA 0..4) they
would accept; 1,446 answers, 98 of them (6.8%) sets. The label table is joined to the ~20 Hz driver-state and vehicle
frames of the same 20 s window, which are summarised per window (means, standard deviations, event fractions) and
completed by a one-hot code of the task asked about. The two prompts of a window share their frames, and a driver's
windows share a calibration, so the driver is the group that is never split between training and test items.

Source: huggingface.co/datasets/ProVoice-proactivity/proactivity_preference_dataset (CC-BY-NC-4.0), fetched by
`python scripts/01_download.py provoice`; `python scripts/02_build.py provoice` writes $SCB_DATA/processed/provoice.npz.
"""
import numpy as np
import pandas as pd

from .native import parse_sets, save_native
from .paths import RAW

K = 5
TASKS = ["Change song", "Provide traffic news", "Provide weather update", "Respond to a phone call",
         "Respond to a text message"]
EMOTIONS = ["neutral", "disgust", "angry", "surprise", "happy", "sad"]
LIGHTS = ["Red", "Yellow"]            # green is the remainder

MEAN = ["eye_ar", "mar", "gaze_score", "blink_rate", "yawn_rate", "perclos", "emotion_prob", "heart_rate",
        "respiratory_rate", "speed_kmh", "speed_limit_kmh", "speed_ratio_max", "speed_ratio_limit", "throttle", "brake",
        "acceleration", "gear", "precipitation", "fog_density", "headway_s"]
STD = ["eye_ar", "gaze_score", "heart_rate", "speed_kmh", "steer", "acceleration"]
ABSMEAN = ["steer"]
FRACTION = ["face_present", "gaze_distracted", "drowsiness_alert", "is_junction", "is_night", "reverse", "hand_brake",
            "headlight", "fog_light", "left_indicator", "right_indicator"]
FRAME_COLUMNS = sorted(set(MEAN + STD + ABSMEAN + FRACTION + ["timestamp", "session_id", "emotion", "traffic_light_state",
                                                              "lead_distance_m"]))


def seconds_of_day(ts):
    """Seconds since midnight of 'HH:MM:SS.mmm' frame stamps or ISO date-times."""
    ts = pd.Series(ts).astype(str)
    iso = ts.str.contains("T")
    out = pd.Series(np.nan, index=ts.index)
    if iso.any():
        dt = pd.to_datetime(ts[iso])
        out[iso] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second + dt.dt.microsecond / 1e6
    if (~iso).any():
        out[~iso] = pd.to_timedelta(ts[~iso]).dt.total_seconds()
    return out.to_numpy()


def read_frames(path, chunk=50_000):
    """Frame columns used by the summary, read in chunks (the file is 0.55 GB)."""
    parts = []
    for c in pd.read_json(path, lines=True, chunksize=chunk, dtype=False):
        c = c[FRAME_COLUMNS].copy()
        c["t"] = seconds_of_day(c.pop("timestamp"))
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def window_summary(frames, windows):
    """One row per window (session_id, window_idx): summary of the frames with start <= t <= end of that session."""
    rows = []
    for sid, w in windows.groupby("session_id"):
        f = frames[frames.session_id == sid]
        w = w.sort_values("start")
        k = np.searchsorted(w.start.to_numpy(), f.t.to_numpy(), side="right") - 1
        inside = (k >= 0) & (f.t.to_numpy() <= w.end.to_numpy()[np.clip(k, 0, None)])
        f, k = f[inside], k[inside]
        for j, g in f.groupby(k):
            lead = g.lead_distance_m.where(g.lead_distance_m >= 0)
            r = {"session_id": sid, "window_idx": int(w.window_idx.iloc[j]), "n_frames": len(g),
                 "hr_available": g.heart_rate.notna().mean(), "lead_fraction": lead.notna().mean(), "lead_distance": lead.mean()}
            r.update({f"{c}_mean": g[c].astype(float).mean() for c in MEAN})
            r.update({f"{c}_std": g[c].astype(float).std(ddof=0) for c in STD})
            r.update({f"{c}_absmean": g[c].astype(float).abs().mean() for c in ABSMEAN})
            r.update({f"{c}_frac": g[c].astype(float).mean() for c in FRACTION})
            r.update({f"emotion_{e}": (g.emotion == e).mean() for e in EMOTIONS})
            r.update({f"light_{c.lower()}": (g.traffic_light_state == c).mean() for c in LIGHTS})
            rows.append(r)
    return pd.DataFrame(rows)


def assemble(labels, frames):
    """Feature matrix, accepted-set mask, ids, groups and feature names for `labels` (one row per prompt)."""
    lab = labels.copy()
    lab["start"] = seconds_of_day(lab.window_start_timestamp)
    lab["end"] = seconds_of_day(lab.window_end_timestamp)
    windows = lab.drop_duplicates(["session_id", "window_idx"])[["session_id", "window_idx", "start", "end"]]
    summary = window_summary(frames, windows)
    lab = lab.merge(summary, on=["session_id", "window_idx"], how="left")
    lab = lab[lab.n_frames.notna()].reset_index(drop=True)          # prompts whose window has no frame cannot be used
    names = [c for c in summary.columns if c not in ("session_id", "window_idx", "n_frames")]
    Xs = lab[names].astype(float)
    Xs = Xs.loc[:, Xs.std(ddof=0) > 0]                              # constants (city, no rain, ...) carry nothing
    Xs = Xs.fillna(Xs.median())                                     # missing heart rate / lead distance: column median
    task = np.stack([(lab.functionname == t).to_numpy(float) for t in TASKS], 1)
    X = np.concatenate([Xs.to_numpy(), task], 1)
    names = list(Xs.columns) + [f"task_{t}" for t in TASKS]
    mask = parse_sets(lab.user_selected_loa.astype(str), K, base=0, sep=";")
    ids = lab.session_id + "|win" + lab.window_idx.astype(str).str.zfill(3) + "p" + lab.prompt_in_window.astype(str)
    return X, mask, ids.to_numpy(), lab.participantid.astype(str).to_numpy(), names


def build_provoice(root=None):
    root = root or RAW / "provoice" / "data"
    labels = pd.read_csv(root / "labels.csv", dtype={"participantid": str})
    frames = read_frames(root / "frames_preprocessed.jsonl")
    X, mask, ids, group, names = assemble(labels, frames)
    print(f"  {len(labels) - len(X)} of {len(labels)} prompts dropped (no frames in their window)")
    return save_native("provoice", X, mask, K, ids, group, extra=dict(feature_names=np.array(names)))


def build_provoice_task():
    """The same answers with the task code as the only feature: what the task alone says about the accepted levels."""
    from .paths import PROCESSED
    z = np.load(PROCESSED / "provoice.npz", allow_pickle=True)
    names = [str(n) for n in z["feature_names"]]
    keep = [i for i, n in enumerate(names) if n.startswith("task_")]
    return save_native("provoice_task", z["X"][:, keep], z["mask_m0"], K, z["item_id"], z["group"],
                       extra=dict(feature_names=np.array([names[i] for i in keep])))


def build_provoice_centred():
    """Driver-state features minus the driver's own mean over all their windows (label-free, as if every driver had driven
    a calibration lap); the task code is left as it is. Removes between-driver offsets from the population models."""
    from .paths import PROCESSED
    z = np.load(PROCESSED / "provoice.npz", allow_pickle=True)
    names = [str(n) for n in z["feature_names"]]
    X, group = z["X"].astype(float).copy(), z["group"]
    state = [i for i, n in enumerate(names) if not n.startswith("task_")]
    for g in np.unique(group):
        rows = group == g
        X[np.ix_(rows, state)] -= X[np.ix_(rows, state)].mean(0)
    return save_native("provoice_centred", X, z["mask_m0"], K, z["item_id"], group, extra=dict(feature_names=np.array(names)))


def build_provoice_known():
    """The personalised setting: a driver's one-hot code is a feature and the split is by 20 s window, so a driver's other
    windows (and their answers) are in training while a window is tested. Neighbouring windows share context, which
    is what a deployed assistant also has from the driver's earlier answers."""
    from .paths import PROCESSED
    z = np.load(PROCESSED / "provoice.npz", allow_pickle=True)
    names = [str(n) for n in z["feature_names"]]
    drivers = sorted(set(z["group"]))
    onehot = np.stack([(z["group"] == d).astype(float) for d in drivers], 1)
    window = np.array([str(i).rsplit("p", 1)[0] for i in z["item_id"]])        # both prompts of a window share its frames
    return save_native("provoice_known", np.concatenate([z["X"], onehot], 1), z["mask_m0"], K, z["item_id"], window,
                       extra=dict(feature_names=np.array(names + [f"driver_{d}" for d in drivers])))
