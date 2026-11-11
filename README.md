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


## System B – Local Features & MinHash
- [x] Extract SIFT Features
- [x] Build Visual Vocabulary with K-Means
- [x] Convert Images to Visual-Word Sets
- [x] Implement MinHash Signatures
- [x] Implement MinHash LSH Banding
- [x] Tune Vocabulary Size & LSH Parameters
- [x] Jaccard Similarity & Containment
- [x] Spatial Verification with RANSAC

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
uv run python -m near_duplicates.download disc21 --scale 50000       # 50,000 further references for the scale point, 8.8 GB
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
| `scale` (DISC21, optional) | References outside every other split; `run --scale` adds them to the index for the 100,000-image point. |

Tune only on `val_*` and `dev_calibration`; report only `test_*` and `dev_report`.
The splits are committed and seeded; do not regenerate them once results exist.
Caltech-101 excludes `Faces_easy`, which holds crops of the `Faces` images.

### Preprocessing, fitting and caching

- `preprocessing.py`: decoding with EXIF orientation and transparency handled; 32 x 32 grayscale for pHash, RGB with a longer side of at most 512 for SIFT.
- `local_features.extract_local_features`: pHash of the image as given and the 300 strongest SIFT keypoints of its 512-pixel copy; edits are applied to the full-resolution image first (`features.py`).
- `fit_image_ids(dataset)`: vocabulary-fitting images excluding validation/test sources and DISC21 development sources; 3,709 images for Caltech-101, 38,013 for DISC21.
- Feature caches and vocabularies live under `data/cache/<dataset>-<settings hash>/`, one `.npz` per image list, so caches made with other extractor settings are never mixed up or overwritten.

Verify local data and run tests:

```sh
uv run python -m near_duplicates.validate caltech101
uv run pytest
```

Expected: 8,709 Caltech images. The evaluation teammate should run the validator with `disc21` after downloading it; DISC21 remains unverified locally.

## Running the experiment

```sh
uv run python -m near_duplicates.run caltech101 --split val --queries 20 --kinds jpeg crop   # smoke test
uv run python -m near_duplicates.run caltech101 --split val    # tune: calibrates and saves the thresholds
uv run python -m near_duplicates.run caltech101 --split test   # report: reuses the validation thresholds
uv run python -m near_duplicates.run disc21 --split val --index-size 10000   # one scaling point
uv run python -m near_duplicates.run disc21 --split dev        # Meta's development queries, micro-average precision
uv run python -m near_duplicates.analyze results/queries_<tag>.csv   # LSH loss and the cascade, no new search
uv run python -m near_duplicates.tune results/queries_<tag>_val.csv --vocabulary 512 2048 8192   # choose LSH, shortlist, vocabulary
uv run python -m near_duplicates.failures results/queries_<tag>_val.csv   # failure mechanisms and a local gallery of misses
uv run python -m near_duplicates.dedup disc21 --index-size 10000     # deduplication by graph clustering
uv run pytest                                                  # tests
```

Four systems are run on the split's targets and distractors under every condition in `transforms.CONDITIONS`, each searched exhaustively and through its LSH index: pHash, visual words, visual words whose top `--top-k` matches are re-ranked by RANSAC inliers (`visual-words+ransac`), and visual words scored by containment instead of Jaccard similarity (`visual-words-containment`); the last two share the visual-word LSH index.
The conditions are single edits at graded strengths, extended until each system fails, and four chains of edits (`transforms.CHAINS`) at levels 1 to 3.
`--index-size` searches a seeded subset of the index that always holds the targets, for the scaling measurements; the vocabulary always comes from the full index.
`--split dev` evaluates on DISC21's own development queries: thresholds from `dev_calibration`, scores on `dev_report`, with micro-average precision computed as in Meta's evaluation code, from the top prediction per query.
Features are cached under `data/cache/` per image list, so a rerun with other LSH or vocabulary settings only repeats the search.
`--help` lists the parameters: vocabulary size, MinHash permutations, bands and rows for both LSH families, the RANSAC shortlist, seed and target false-match rate.
Besides the per-query CSV, each run writes a summary CSV, `costs_<tag>.csv` (index memory, candidates and milliseconds per query) and figures.

Every run writes `results/queries_<tag>.csv` with one row per query, system and mode, in the shared schema of `metrics.RESULT_COLUMNS`:

| Column | Meaning |
| --- | --- |
| `system`, `mode` | `phash` or `visual-words`; `exhaustive` or `lsh` |
| `kind`, `strength` | the transformation and its parameter |
| `query_id`, `true_id` | the source image; the indexed image it should match, empty for distractors |
| `predicted`, `score` | the top result and its score, higher meaning more similar; empty and `-inf` without LSH candidates |
| `candidates`, `true_is_candidate` | candidates searched (the whole index when exhaustive); whether LSH proposed the true match |
| `true_similarity` | bit agreement 1 - d/64 for pHash, Jaccard for visual words, RANSAC inliers for `visual-words+ransac`, containment for `visual-words-containment`, between query and true match |
| `seconds` | search time after feature extraction: scoring the whole index, or retrieving and scoring the LSH candidates |

Any new system or combination that writes this schema can reuse `metrics.summarize` and the plots.

`analyze` reads a per-query CSV and repeats no search.
It compares measured LSH recall with the recall the s-curve predicts from each target's true similarity (`lsh_loss_<tag>.csv`).
It also combines pHash with each other system: the jointly calibrated cascade, score fusion and the best possible combination, all calibrated on half of the source images and evaluated on the other half at one false-match rate (`combinations_*`); on the development queries, calibrated on `dev_calibration` and evaluated on `dev_report`.

`tune` computes every validation query's similarity to every indexed image once and predicts, for every LSH setting, recall at the threshold and the candidate fraction; for spatial verification also whether the true image enters the shortlist, and for each vocabulary size its Davies-Bouldin index (`tuning_<tag>.csv`).
It checks its own predictions against the run's measured values, so the error of the method is printed beside its choices.

`failures` tabulates per condition what each representation keeps of the true pair (Hamming distance, keypoints, Jaccard similarity, ratio-test matches, inliers; `mechanisms_<tag>.csv`), and draws sampled misses at each system's breaking point under `data/failures/`, with attributions, so the images never enter git.

`dedup` plants transformed copies of random indexed images, joins the catalogue with itself through both LSH indexes, verifies visual-word pairs with RANSAC, and clusters the graph of pairs above a similarity threshold with connected components and with Louvain, scored by the adjusted Rand index, pairwise precision and recall, and modularity.

The summary adds recall@1, recall at the false-match threshold and the false-match rate per condition, each with a 95% Wilson interval.
Rates pooled over conditions (`overall_<tag>.csv`, and the combinations) carry 95% intervals from resampling source images, because each source is queried under every condition and its queries are not independent.

## Repository layout

- `src/near_duplicates/`: the package; all shared code lives here.
  - `datasets.py`: data locations, splits and DISC21 ground truth.
  - `download.py`: the download command above.
  - `preprocessing.py`: preprocessing presets shared by catalogue images and queries.
  - `validate.py`: read-only split, image and DISC21 ground-truth audit.
  - `phash.py`: the 64-bit DCT pHash and exact Hamming distances.
  - `local_features.py`: joint pHash and SIFT extraction from one decoded image.
  - `transforms.py`: the transformation grid and the seeded transformations.
  - `features.py`: edits at full resolution, extraction in parallel, and the per-list feature caches.
  - `visual_words.py`: k-means vocabularies, word assignment and the Davies-Bouldin diagnostic.
  - `vocabulary.py`: the cached vocabulary fitted on the fitting pool, and visual-word sets.
  - `minhash.py`: MinHash signatures from exact random permutations.
  - `lsh.py`: bit sampling, band keys, the banded index for both LSH families and their s-curves.
  - `spatial.py`: ratio-test matching and RANSAC homography verification.
  - `systems.py`: the systems behind one interface, and the search loop.
  - `metrics.py`: the results schema, threshold calibration and recall with intervals.
  - `cascade.py`: the cascade, score fusion and the best possible combination.
  - `dedup.py`: planted copies, the LSH self-join, verified edges and graph clustering.
  - `tune.py`: LSH, shortlist and vocabulary choices predicted from one validation run.
  - `failures.py`: failure mechanisms and the gallery of misses.
  - `plots.py`, `run.py`, `analyze.py`: figures and the commands above.
- `splits/`: the fixed splits, one image ID per line.
- `tests/`: pytest suite; tests that need Caltech-101 skip when it is not downloaded.
- `docs/findings.md`: reference findings on validation data, and what each seat inherits from them.
- `data/`: downloaded images and feature caches; gitignored.
- `results/`: figures and run summaries; per-query CSVs are gitignored.

## Data licence

DISC21 (Meta, Image Similarity Challenge 2021): <https://ai.meta.com/datasets/disc21-dataset/>.
Its licence limits use to research on detecting manipulated media and forbids redistributing the dataset.
Never commit images; the code downloads them from Meta's servers.
The report may show at most 100 DISC21 images, each with the attribution from Meta's attribution CSV.

Douze et al., "The 2021 Image Similarity Dataset and Challenge", 2021, <https://arxiv.org/abs/2106.09672>.

Caltech-101 (CaltechDATA, DOI 10.22002/D1.20086) is CC BY 4.0; cite Fei-Fei, Fergus and Perona, "One-shot learning of object categories", TPAMI 2006.
