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