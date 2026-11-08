# Detecting-Duplicate-and-Manipulated-Images

## Setup & Data

- [x] Project Setup & Dependencies
- [x] Download Caltech101 (everyone needs their own copy)
- [ ] Download DISC21 Dataset (only one person do this)
- [x] Image Preprocessing
- [x] Feature Caching & Data Loading
- [x] Train, Validation & Test Splits

## System A – Perceptual Hashing
- [x] Implement pHash (DCT-Based)
- [x] Validate pHash Implementation
- [x] Implement Bit-Sampling LSH
- [x] Tune LSH Parameters (b, r)
- [x] Hamming Distance & Candidate Verification


```sh
PYTHONPATH=src uv run python -m near_duplicates.system_a caltech101 --output results/system_a_caltech101.json
```

This tunes `(b, r)` and the Hamming threshold on the fixed validation splits;
test and DISC21 report queries remain reserved. The full Caltech calibration selected
`b=32`, `r=8`, Hamming threshold `10`: validation Recall@1 **79.44%**, precision
**87.26%** across 5,000 transformed target/distractor queries. See the
[run summary](results/system_a_caltech101.json) and method notes for tradeoffs
and crop/rotation limitations.

## System B – Local Features & MinHash
- [ ] Extract SIFT Features
- [ ] Build Visual Vocabulary with K-Means
- [ ] Convert Images to Visual-Word Sets
- [ ] Implement MinHash Signatures
- [ ] Implement MinHash LSH Banding
- [ ] Tune Vocabulary Size & LSH Parameters
- [ ] Jaccard Similarity & Containment
- [ ] Spatial Verification with RANSAC


## Testing & Evaluation
- [ ] Generate Transformed Image Queries
- [ ] Generate Distractor Queries
- [ ] Implement Exhaustive Search Baselines
- [ ] Evaluate Recall@1 & Precision
- [ ] Compare LSH vs. Exhaustive Search
- [ ] Evaluate LSH S-Curves
- [ ] Analyze Failure Cases

## Scalability & Clustering
- [ ] Benchmark Query Time & Memory
- [ ] Test Scalability (10K, 50K, 100K–200K Images)
- [ ] Detect Near-Duplicate Pairs with LSH Self-Join
- [ ] Build Image Similarity Graph
- [ ] Compare Louvain vs. Connected Components
- [ ] Evaluate Clustering Quality (ARI & Modularity)

## Final Experiments & Results
- [ ] Compare pHash vs. Visual Words
- [ ] Analyze Performance by Transformation Strength
- [ ] Run DISC21 Development Evaluation
- [ ] Perform Parameter Ablation Studies
- [ ] Generate Final Figures & Tables
- [ ] Summarize Findings & Limitations


## Report & Submission
- [ ] Write Introduction & Research Questions
- [ ] Describe Dataset & Preprocessing
- [ ] Explain Methods & Algorithms
- [ ] Present Experimental Results
- [ ] Discuss Failures & Trade-offs
- [ ] Write Conclusion
- [ ] Finalize Contribution Table
- [ ] Clean & Document Reproducible Notebook
- [ ] Submit Report & Notebook

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync --locked
```

## Data

Caltech-101 is the development set: everyone runs on it locally.
DISC21 is the evaluation set for the reported results; one machine downloads it.

```sh
uv run python -m near_duplicates.download caltech101                 # 130 MB, under a minute
uv run python -m near_duplicates.download disc21 --limit 100         # smoke test, 200 images
uv run python -m near_duplicates.download disc21                     # about 72,500 images, roughly 12 GB
```

The DISC21 download fetches single images out of Meta's 8.8 GB zip archives with HTTP range requests, so only the images in the splits are transferred.
Reruns skip finished images, so an interrupted download resumes.
Set `NEAR_DUPLICATES_DATA` to keep the data outside the repository, for example on shared HPC storage.

In code, use the fixed splits instead of listing files:

```python
from near_duplicates.datasets import load_split, image_path, dev_ground_truth

index = load_split("caltech101", "index")                 # images to index
targets = load_split("caltech101", "val_targets")         # indexed images to transform into queries
distractors = load_split("caltech101", "val_distractors") # non-indexed images, for false matches
path = image_path("caltech101", index[0])
```

| Split | Contents |
| --- | --- |
| `index` | Images in the catalogue. DISC21: the 10,000 references behind the positive development queries plus random others, 50,000 in total. |
| `val_targets`, `test_targets` | 500 and 2,000 indexed images to transform into queries with a known answer. |
| `val_distractors`, `test_distractors` | 500 and 2,000 images outside the index, transformed the same way, to measure false matches. |
| `dev_calibration`, `dev_report` (DISC21) | Half each of 10,000 matched and 10,000 unmatched development queries; `dev_ground_truth()` gives the answers. |

Tune only on `val_*` and `dev_calibration`; report only `test_*` and `dev_report`.
The splits are committed and seeded; do not regenerate them once results exist.
Caltech-101 excludes `Faces_easy`, which holds crops of the `Faces` images.

### Preprocessing, fitting and caching

- `preprocessing.py`: grayscale `uint8` images; 32 × 32 for pHash, maximum side 1,024 for SIFT (aspect ratio preserved). Use `iter_batches()` for loading.
- `fit_image_ids(dataset)`: vocabulary-fitting images excluding validation/test sources; 3,709 images for Caltech-101. Use seed 2807 for sampling and fitting.
- `FeatureCache`: saves numeric feature arrays under `data/cache/`. Include extractor settings/version and vocabulary identity in cache parameters. Feature extraction follows in Systems A/B.

Verify local data and run tests:

```sh
uv run python -m near_duplicates.validate caltech101
uv run python -m unittest discover -s tests -v
```

Expected: 8,709 Caltech images. The evaluation teammate should run the validator with `disc21` after downloading it; DISC21 remains unverified locally.

## Running the experiment

```sh
uv run python -m near_duplicates.run caltech101 --split val --queries 20 --kinds jpeg crop   # smoke test
uv run python -m near_duplicates.run caltech101 --split val    # tune: calibrates and saves the thresholds
uv run python -m near_duplicates.run caltech101 --split test   # report: reuses the validation thresholds
uv run python -m near_duplicates.run disc21 --split val --index-size 10000   # one scaling point
uv run python -m near_duplicates.run disc21 --split dev        # Meta's development queries, micro-average precision
uv run pytest                                                  # tests
```

Both systems are run on the split's targets and distractors under every condition in `transforms.CONDITIONS`, each searched exhaustively and through its LSH index.
The conditions are single edits at graded strengths, extended until each system fails, and four chains of edits (`transforms.CHAINS`) at levels 1 to 3.
`--index-size` searches a seeded subset of the index that always holds the targets, for the scaling measurements; the vocabulary always comes from the full index.
`--split dev` evaluates on DISC21's own development queries: thresholds from `dev_calibration`, scores on `dev_report`, with micro-average precision computed as in Meta's evaluation code, from the top prediction per query.
Features are cached under `data/cache/` per image list, so a rerun with other LSH or vocabulary settings only repeats the search.
`--help` lists the parameters: vocabulary size, MinHash permutations, bands and rows for both LSH families, seed and target false-match rate.
Besides the per-query CSV, each run writes a summary CSV, `costs_<tag>.csv` (index memory, candidates and milliseconds per query) and figures.

Every run writes `results/queries_<tag>.csv` with one row per query, system and mode, in the shared schema of `metrics.RESULT_COLUMNS`:

| Column | Meaning |
| --- | --- |
| `system`, `mode` | `phash` or `visual-words`; `exhaustive` or `lsh` |
| `kind`, `strength` | the transformation and its parameter |
| `query_id`, `true_id` | the source image; the indexed image it should match, empty for distractors |
| `predicted`, `score` | the top result and its score, higher meaning more similar; empty and `-inf` without LSH candidates |
| `candidates`, `true_is_candidate` | candidates searched (the whole index when exhaustive); whether LSH proposed the true match |
| `true_similarity` | bit agreement 1 - d/64 for pHash, Jaccard for visual words, between query and true match |
| `seconds` | search time after feature extraction: scoring the whole index, or retrieving and scoring the LSH candidates |

Any new system or combination (the cascade, spatial verification) that writes this schema can reuse `metrics.summarize` and the plots.
The summary adds recall@1, recall at the false-match threshold and the false-match rate, each with a 95% Wilson interval.

## Repository layout

- `src/near_duplicates/`: the package; all shared code lives here.
  - `datasets.py`: data locations, splits and DISC21 ground truth.
  - `download.py`: the download command above.
  - `preprocessing.py`: preprocessing presets and streaming image/batch loading.
  - `cache.py`: per-image feature caches with metadata and source-content invalidation.
  - `validate.py`: read-only split, image and DISC21 ground-truth audit.
  - `phash.py`: cached 64-bit DCT pHashes and exact Hamming distances.
  - `system_a.py`: validation query probes, parameter sweep and calibration command.
  - `transforms.py`: the transformation grid and the seeded transformations.
  - `features.py`: preprocessing, pHash, SIFT and the feature cache.
  - `lsh.py`: MinHash, bit sampling, band keys, the banded index and the theoretical s-curves; System A's seeded bit-sampling tables with full-hash candidate verification.
  - `vocabulary.py`: the k-means vocabulary and visual-word sets.
  - `systems.py`: both systems behind one interface, and the search loop.
  - `metrics.py`: the results schema, threshold calibration and recall with intervals.
  - `plots.py`, `run.py`: figures and the experiment command above.
- `splits/`: the fixed splits, one image ID per line.
- `tests/`: pytest suite, including the synthetic setup/data and System A checks; tests that need Caltech-101 skip when it is not downloaded.
- `docs/system_a.md`: method, theory, validation and usage for System A.
- `data/`: downloaded images and feature caches; gitignored.
- `results/`: figures and run summaries; per-query CSVs are gitignored.

## Data licence

DISC21 (Meta, Image Similarity Challenge 2021): <https://ai.meta.com/datasets/disc21-dataset/>.
Its licence limits use to research on detecting manipulated media and forbids redistributing the dataset.
Never commit images; the code downloads them from Meta's servers.
The report may show at most 100 DISC21 images, each with the attribution from Meta's attribution CSV.

Douze et al., "The 2021 Image Similarity Dataset and Challenge", 2021, <https://arxiv.org/abs/2106.09672>.

Caltech-101 (CaltechDATA, DOI 10.22002/D1.20086) is CC BY 4.0; cite Fei-Fei, Fergus and Perona, "One-shot learning of object categories", TPAMI 2006.
