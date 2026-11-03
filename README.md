# Detecting-Duplicate-and-Manipulated-Images

## Setup & Data

- Project Setup & Dependencies
- Download DISC21 Dataset
- Image Preprocessing
- Feature Caching & Data Loading
- Train, Validation & Test Splits

## System A – Perceptual Hashing
- Implement pHash (DCT-Based)
- Validate pHash Implementation
- Implement Bit-Sampling LSH
- Tune LSH Parameters (b, r)
- Hamming Distance & Candidate Verification

## System B – Local Features & MinHash
- Extract SIFT Features
- Build Visual Vocabulary with K-Means
- Convert Images to Visual-Word Sets
- Implement MinHash Signatures
- Implement MinHash LSH Banding
- Tune Vocabulary Size & LSH Parameters
- Jaccard Similarity & Containment
- Spatial Verification with RANSAC


## Testing & Evaluation
- Generate Transformed Image Queries
- Generate Distractor Queries
- Implement Exhaustive Search Baselines
- Evaluate Recall@1 & Precision
- Compare LSH vs. Exhaustive Search
- Evaluate LSH S-Curves
- Analyze Failure Cases

## Scalability & Clustering
- Benchmark Query Time & Memory
- Test Scalability (10K, 50K, 100K–200K Images)
- Detect Near-Duplicate Pairs with LSH Self-Join
- Build Image Similarity Graph
- Compare Louvain vs. Connected Components
- Evaluate Clustering Quality (ARI & Modularity)

## Final Experiments & Results
- Compare pHash vs. Visual Words
- Analyze Performance by Transformation Strength
- Run DISC21 Development Evaluation
- Perform Parameter Ablation Studies
- Generate Final Figures & Tables
- Summarize Findings & Limitations


## Report & Submission
- Write Introduction & Research Questions
- Describe Dataset & Preprocessing
- Explain Methods & Algorithms
- Present Experimental Results
- Discuss Failures & Trade-offs
- Write Conclusion
- Finalize Contribution Table
- Clean & Document Reproducible Notebook
- Submit Report & Notebook

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync
```

## Repository layout

- `src/near_duplicates/`: the package; all shared code lives here.
- `data/`: downloaded images and feature caches; gitignored.
- `results/`: figures and run summaries; per-query CSVs are gitignored.

## Data licence

DISC21 (Meta, Image Similarity Challenge 2021): <https://ai.meta.com/datasets/disc21-dataset/>.
Its licence limits use to research on detecting manipulated media and forbids redistributing the dataset.
Never commit images; the code downloads them from Meta's servers.
The report may show at most 100 DISC21 images, each with the attribution from Meta's attribution CSV.

Douze et al., "The 2021 Image Similarity Dataset and Challenge", 2021, <https://arxiv.org/abs/2106.09672>.
