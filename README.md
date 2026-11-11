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

## Repository layout

- `src/near_duplicates/`: the package; all shared code lives here.
  - `datasets.py`: data locations, splits and DISC21 ground truth.
  - `download.py`: the download command above.
  - `preprocessing.py`: preprocessing presets and streaming image/batch loading.
  - `cache.py`: per-image feature caches with metadata and source-content invalidation.
  - `validate.py`: read-only split, image and DISC21 ground-truth audit.
  - `phash.py`: cached 64-bit DCT pHashes and exact Hamming distances.
  - `lsh.py`: seeded bit-sampling tables and full-hash candidate verification.
  - `system_a.py`: validation query probes, parameter sweep and calibration command.
- `splits/`: the fixed splits, one image ID per line.
- `tests/`: synthetic setup/data and System A checks; no dataset download needed.
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
