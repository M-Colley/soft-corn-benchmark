"""Turn each raw multi-annotator dataset into frozen-encoder features plus per-item ratings,
then derive set-valued labels.

For every item, m annotators are drawn once (fixed seed) from its rating pool. The
set-valued label is the set of levels those m annotators chose; the remaining annotators are
kept as held-out ratings that never enter any training or selection signal.

Output: $SCB_DATA/processed/<name>.npz with
    X (n, d) float32 features, K, ratings_flat / ratings_ptr (all ratings, 0-indexed levels),
    and for each m in M_VALUES: mask_m{m} (n, K) bool, ho_m{m} (n, K) held-out counts,
    valid_m{m} (n,) items with at least m ratings.

    python scripts/02_build.py [names...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import PROCESSED, RAW  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

M_VALUES = (2, 3, 5)
ANNOTATOR_SEED = 12345
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
IMAGE_MODEL = ("ViT-B-16", "laion2b_s34b_b88k")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_st = None


def embed_text(texts):
    global _st
    if _st is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(TEXT_MODEL, device=DEVICE)
    return _st.encode(list(texts), batch_size=128, show_progress_bar=True, convert_to_numpy=True,
                      normalize_embeddings=False).astype(np.float32)


def embed_images(paths):
    import open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(IMAGE_MODEL[0], pretrained=IMAGE_MODEL[1],
                                                                 device=DEVICE)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + 64]]).to(DEVICE)
            feats.append(model.encode_image(batch).float().cpu().numpy())
            print(f"  images {i + len(batch)}/{len(paths)}", end="\r")
    return np.concatenate(feats).astype(np.float32)


def sets_from_ratings(ratings, K, m, rng):
    n = len(ratings)
    mask = np.zeros((n, K), bool)
    ho = np.zeros((n, K), np.int32)
    valid = np.zeros(n, bool)
    for i, r in enumerate(ratings):
        if len(r) < m:
            continue
        valid[i] = True
        perm = rng.permutation(len(r))
        mask[i, r[perm[:m]]] = True
        np.add.at(ho[i], r[perm[m:]], 1)
    return mask, ho, valid


def save(name, X, ratings, K, extra=None):
    ratings = [np.asarray(r, np.int64) for r in ratings]
    assert all(((r >= 0) & (r < K)).all() for r in ratings)
    ptr = np.concatenate([[0], np.cumsum([len(r) for r in ratings])])
    out = dict(X=X.astype(np.float32), K=K, ratings_flat=np.concatenate(ratings), ratings_ptr=ptr)
    for m in M_VALUES:
        rng = np.random.default_rng(ANNOTATOR_SEED + m)
        out[f"mask_m{m}"], out[f"ho_m{m}"], out[f"valid_m{m}"] = sets_from_ratings(ratings, K, m, rng)
    if extra:
        out.update(extra)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / f"{name}.npz", **out)
    n_r = np.diff(ptr)
    print(f"[{name}] n={len(ratings)} d={X.shape[1]} K={K} ratings/item median={np.median(n_r):.0f} "
          f"valid(m=3)={out['valid_m3'].sum()}")


def build_mhs():
    # Kennedy et al. (2020). Item: comment; target: the 5-level `sentiment` item (0-4).
    df = pd.read_parquet(RAW / "mhs" / "measuring-hate-speech.parquet",
                         columns=["comment_id", "annotator_id", "sentiment", "text"])
    df = df.dropna(subset=["sentiment"])
    g = df.groupby("comment_id")
    items = g.size()
    items = items[items >= 2].index
    sub = df[df.comment_id.isin(items)]
    ratings = sub.groupby("comment_id")["sentiment"].apply(lambda s: s.astype(int).to_numpy())
    texts = sub.groupby("comment_id")["text"].first().loc[ratings.index]
    save("mhs", embed_text(texts.tolist()), ratings.tolist(), 5,
         {"item_id": ratings.index.to_numpy()})


def build_wikitox(n_items=40000):
    # Wulczyn et al. (2017). toxicity_score in {-2..2} -> 0 (very toxic) .. 4 (very healthy).
    ann = pd.read_csv(RAW / "wikitox" / "toxicity_annotations.tsv", sep="\t")
    com = pd.read_csv(RAW / "wikitox" / "toxicity_annotated_comments.tsv", sep="\t", index_col=0)
    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(com.index.to_numpy(), size=n_items, replace=False))
    ann = ann[ann.rev_id.isin(ids)]
    ratings = ann.groupby("rev_id")["toxicity_score"].apply(lambda s: (s.astype(int) + 2).to_numpy())
    text = com.loc[ratings.index, "comment"].str.replace("NEWLINE_TOKEN", " ").str.replace("TAB_TOKEN", " ")
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    save("wikitox", embed_text(text.tolist()), ratings.tolist(), 5,
         {"item_id": ratings.index.to_numpy()})


def build_humicroedit():
    # Hossain et al. (2020). Five funniness grades 0-3 per edited headline. Features: embedding
    # of the edited headline and its difference to the original headline.
    d = RAW / "humicroedit" / "subtask-1"
    df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))], ignore_index=True)
    df = df.drop_duplicates("id")
    ratings = [np.array([int(c) for c in str(g)]) for g in df["grades"]]
    e_edit = embed_text(df["edited"].tolist())
    e_orig = embed_text(df["headline"].tolist())
    save("humicroedit", np.concatenate([e_edit, e_edit - e_orig], 1), ratings, 4,
         {"item_id": df["id"].to_numpy()})


def build_cb():
    # de Marneffe et al. (2019). 7-point projection judgments (-3..3) -> 0..6. Features:
    # embedding of context + target sentence, and of the prompt (the embedded complement).
    df = pd.read_csv(RAW / "cb" / "CommitmentBank-All.csv")
    g = df.groupby("uID", sort=True)
    ratings = g["Answer"].apply(lambda s: (s.astype(int) + 3).to_numpy())
    first = g.first().loc[ratings.index]
    ctx = (first["Context"].fillna("") + " " + first["Target"].fillna("")).str.strip()
    X = np.concatenate([embed_text(ctx.tolist()), embed_text(first["Prompt"].fillna("").tolist())], 1)
    save("cb", X, ratings.tolist(), 7, {"item_id": ratings.index.to_numpy()})


def build_scut():
    # Liang et al. (2018). 60 raters x 5,500 faces on 1-5 -> 0..4. Features: OpenCLIP image embedding.
    root = RAW / "scut" / "SCUT-FBP5500_v2"
    df = pd.read_excel(root / "All_Ratings.xlsx")
    ratings = df.groupby("Filename")["Rating"].apply(lambda s: (s.astype(int) - 1).to_numpy())
    paths = [str(root / "Images" / f) for f in ratings.index]
    save("scut", embed_images(paths), ratings.tolist(), 5, {"item_id": ratings.index.to_numpy()})


BUILDERS = {"mhs": build_mhs, "wikitox": build_wikitox, "humicroedit": build_humicroedit,
            "cb": build_cb, "scut": build_scut}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(BUILDERS):
        BUILDERS[n]()
