"""Further multi-annotator ordinal datasets (FUTURE_WORK item 6): items, texts and ratings in one fixed order, shared by
scripts/02_build.py (features) and ordbench.finetune (fine-tuning inputs) so both see the same items.

    vader_*      Hutto & Gilbert (2014), VADER ground truth: sentiment intensity of four text corpora rated on -4..+4 (K = 9)
                 by at least 20 raters per item (MIT; github.com/cjhutto/vaderSentiment)
    convabuse    Cercas Curry, Abercrombie & Rieser (2021): severity of abuse in chatbot conversations, five ordered
                 levels (not abusive < ambiguous < mildly < strongly < very strongly abusive), 2-7 annotators per
                 example (CC-BY-4.0; github.com/amandacurry/convabuse)

Raw files live under $SCB_DATA/raw (scripts/01_download.py vader convabuse). They are third-party data: they are only read
as text, never executed.
"""
import json

import numpy as np
import pandas as pd

from .paths import RAW

VADER = {"vader_tweets": "tweets", "vader_nyt": "nytEditorialSnippets", "vader_movie": "movieReviewSnippets",
         "vader_amazon": "amazonReviewSnippets"}
VADER_K = 9
CONVABUSE_K = 5
SEVERITY_COLUMNS = ["is_abuse.1", "is_abuse.0", "is_abuse.-1", "is_abuse.-2", "is_abuse.-3"]      # severity 0..4
CONVABUSE_KEY = ["conv_id", "prev_agent", "prev_user", "agent", "user", "bot"]
EXTRA = list(VADER) + ["convabuse"]


def _read(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def vader_items(name):
    """(ids, texts, ratings) of one VADER corpus; ratings are arrays of levels 0..8 (rating + 4)."""
    root = RAW / "vader" / "extracted" / "hutto_ICWSM_2014"
    corpus = VADER[name]
    text = {}
    for line in _read(root / f"{corpus}_GroundTruth.txt"):
        parts = line.split("\t", 2)
        if len(parts) == 3:
            text[parts[0].strip()] = parts[2].strip()
    ids, texts, ratings = [], [], []
    for line in _read(root / f"{corpus}_anonDataRatings.txt"):
        parts = line.split("\t")
        if len(parts) < 4 or parts[0].strip() not in text:
            continue
        r = np.rint(np.asarray(json.loads(parts[3]), float) + 4).astype(np.int64)
        if len(r) and r.min() >= 0 and r.max() < VADER_K:
            ids.append(parts[0].strip())
            texts.append(text[parts[0].strip()])
            ratings.append(r)
    return np.array(ids), texts, ratings


def convabuse_items():
    """(ids, (user utterance, whole exchange) text pairs, ratings): one item per distinct exchange, one rating per
    annotator (a repeated judgement by the same annotator keeps its first answer), severity 0..4."""
    df = pd.read_csv(RAW / "convabuse" / "ConvAbuseEMNLPfull.csv")
    df["level"] = df[SEVERITY_COLUMNS].to_numpy().argmax(1)
    df = df.drop_duplicates(CONVABUSE_KEY + ["annotator_id"])
    g = df.groupby(CONVABUSE_KEY, sort=False, dropna=False)
    ratings = g["level"].apply(lambda s: s.to_numpy(np.int64))
    first = g.first().reset_index().fillna("")
    texts = [(str(u), " ".join(str(x) for x in (pa, pu, a, u)).strip())
             for pa, pu, a, u in zip(first.prev_agent, first.prev_user, first.agent, first.user)]
    return np.arange(len(first)), texts, list(ratings.to_numpy())
