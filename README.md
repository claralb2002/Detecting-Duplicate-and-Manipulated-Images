# Near-duplicate image detection at scale

Group project for DTU course 02807 Computational Tools for Data Science.

We detect edited copies of known images, such as crops, rotations, filters and overlays, among 50,000 reference images of Meta's DISC21 dataset.
Two representations are compared: a global perceptual hash (pHash) and local SIFT features turned into visual words, optionally verified with RANSAC.
Both are searched exhaustively and through our own locality-sensitive hashing (bit sampling for pHash, MinHash for visual words), and the same indexes group duplicates within a catalogue by graph clustering.

## Results

Test recall at a 1% false-match rate, over 52 edit conditions and 2,000 source images:

| System | Exhaustive | LSH |
| --- | --- | --- |
| pHash | 0.482 | 0.510 |
| Visual words | 0.610 | 0.609 |
| Visual words + RANSAC | 0.832 | 0.790 |
| Cascade (pHash, then visual words + RANSAC) | 0.887 | 0.867 |

- At 500,000 indexed images, LSH cuts the verified search from 76.2 to 18.8 ms per query.
- On Meta's own development queries, the verified system's recall falls to 0.471.
- Deduplication with verified edges recovers planted groups of copies with an adjusted Rand index of 0.912.

## Running it

Everything is in [`notebook.ipynb`](notebook.ipynb), which imports nothing else from this repository.
It downloads the data, rebuilds and checks the fixed splits, and runs every experiment.

```sh
uv sync          # Python 3.12 and pinned dependencies
```

Then open the notebook, set `DATASET` in the configuration cell, and run all cells:

- `"caltech101"` runs every experiment on Caltech-101 (a 137 MB download, about 2 to 3 hours).
- `"disc21"` runs them at full size: about 21 GB of DISC21 (about 90 GB with the largest scale points) and about 10.5 hours on 32 threads.
- `QUERY_LIMIT = 100` turns either into a smoke test of a few minutes.

Features, vocabularies and thresholds are cached under `data/notebook-cache/`, so an interrupted run resumes where it stopped.
Results and figures are written to `results/notebook/`.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebook.ipynb` | The self-contained reproduction notebook |
| `src/near_duplicates/` | The reference pipeline as a package, with command-line tools; see [`docs/development.md`](docs/development.md) |
| `splits/` | The fixed, seeded splits, one image ID per line |
| `tests/` | The pytest suite for the package (`uv run pytest`) |
| `docs/findings.md` | Findings recorded while developing the pipeline |

## Data licence

DISC21 (Douze et al., [The 2021 Image Similarity Dataset and Challenge](https://arxiv.org/abs/2106.09672)) may be used only for research on detecting manipulated media and may not be redistributed.
The code downloads the images from [Meta's servers](https://ai.meta.com/datasets/disc21-dataset/); never commit them.

Caltech-101 (CaltechDATA, DOI 10.22002/D1.20086) is CC BY 4.0; cite Fei-Fei, Fergus and Perona, "One-shot learning of object categories", TPAMI 2006.
