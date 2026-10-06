"""Download the public multi-annotator ordinal datasets used by the benchmark.

Every source is the dataset's official release (or the canonical Hugging Face
copy). Raw files land in $SCB_DATA/raw/<name>/ and are never modified afterwards.

    python scripts/01_download.py            # all datasets
    python scripts/01_download.py mhs cb     # a subset
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import RAW  # noqa: E402  (sets HF_HOME before huggingface_hub loads)

import requests  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


def _get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        print(f"  GET {url} (attempt {attempt + 1})")
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.replace(dest)
            return dest
        except requests.exceptions.RequestException as e:
            print(f"  failed: {type(e).__name__}")
    raise RuntimeError(f"could not download {url}")


def mhs():
    # Kennedy et al. (2020), Measuring Hate Speech corpus, CC-BY-4.0.
    out = RAW / "mhs"
    out.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("ucberkeley-dlab/measuring-hate-speech", "measuring-hate-speech.parquet",
                        repo_type="dataset", local_dir=out)
    print("  ->", p)


def wiki_toxicity():
    # Wulczyn, Thain & Dixon (2017), Wikipedia Talk Labels: Toxicity, CC0 (figshare 4563973).
    out = RAW / "wikitox"
    _get("https://ndownloader.figshare.com/files/7394542", out / "toxicity_annotated_comments.tsv")
    _get("https://ndownloader.figshare.com/files/7394539", out / "toxicity_annotations.tsv")


def humicroedit():
    # Hossain et al. (2019/2020), SemEval-2020 Task 7 (Humicroedit), five 0-3 grades per edit.
    out = RAW / "humicroedit"
    for split in ["train", "validation", "test"]:
        hf_hub_download("tasksource/humicroedit", f"subtask-1/{split}-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=out)


def commitmentbank():
    # de Marneffe, Simons & Tonhauser (2019), CommitmentBank, per-participant 7-point responses.
    out = RAW / "cb"
    _get("https://raw.githubusercontent.com/mcdm/CommitmentBank/master/CommitmentBank-All.csv",
         out / "CommitmentBank-All.csv")


def scut_fbp5500():
    # Liang et al. (2018), SCUT-FBP5500 v2: 5,500 faces x 60 raters on a 1-5 scale.
    # Official Google Drive link from github.com/HCIILAB/SCUT-FBP5500-Database-Release.
    import gdown
    out = RAW / "scut"
    out.mkdir(parents=True, exist_ok=True)
    z = out / "SCUT-FBP5500_v2.1.zip"
    if not z.exists():
        gdown.download(id="1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf", output=str(z), quiet=False)
    with zipfile.ZipFile(z) as f:
        f.extractall(out)
    print("  ->", [p.name for p in out.iterdir()])


ALL = {"mhs": mhs, "wikitox": wiki_toxicity, "humicroedit": humicroedit,
       "cb": commitmentbank, "scut": scut_fbp5500}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(f"[{n}]")
        ALL[n]()
