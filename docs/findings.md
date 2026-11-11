# Reference findings

Hypotheses tested with the reference pipeline before the seats start, so each seat knows what to expect and where the open questions are.
All numbers are validation numbers: tune on them, never report them as test results.
DISC21 runs use 50,000 indexed references, 500 targets and 500 distractors under each of the 52 conditions, and a 1% false-match target.

| Run | Command |
| --- | --- |
| MVP settings | `run disc21 --split val` (b, r = 16, 8 for pHash; 128 permutations, 64, 2 for MinHash) |
| Tuned LSH and RANSAC | `run disc21 --split val --bit-bands 24 --bit-rows 12 --num-perm 256 --minhash-bands 85 --minhash-rows 3` |
| Combinations and LSH loss | `analyze results/queries_<tag>.csv` |
| Deduplication | `dedup disc21 --index-size 10000 --groups 1000 --copies 3` |

## 1. The s-curve predicts LSH recall (RQ2): confirmed

Predicted LSH recall@1 is the mean over targets of [exhaustive search correct] x P(candidate | true similarity), with the exact bit-sampling curve for pHash and the MinHash s-curve for visual words.
With the MVP settings, mean recall@1 over the 52 conditions is 0.600 measured against 0.598 predicted for pHash, and 0.717 against 0.713 for visual words; per condition the two agree within about 0.04.
With the tuned settings, pHash is 0.588 measured against 0.586 predicted.
So the measured true-pair similarity is enough to choose (b, r) without rerunning the search.

## 2. LSH loss at the operating threshold is zero for the MVP settings, so LSH can be far more selective

With the MVP settings, LSH loses recall@1 only on true pairs whose similarity already lies below the 1% decision threshold, so recall at the threshold is the same with and without LSH.
Moving the s-curve's steep part to just below the decision threshold (pHash accepts d <= 7, visual words J > 0.29) cuts the candidates:

| System | Settings | Candidates | ms per query, LSH | ms per query, scan | Recall at threshold, LSH |
| --- | --- | --- | --- | --- | --- |
| pHash | b, r = 16, 8 | 6.8% | 0.173 | 0.047 | 0.480 |
| pHash | b, r = 24, 12 | 0.8% | 0.034 | 0.049 | 0.480 |
| Visual words | 128 perm, b, r = 64, 2 | 19.6% | 4.00 | 3.68 | 0.517 |
| Visual words | 256 perm, b, r = 85, 3 | 2.2% | 0.58 | 3.65 | 0.509 |

With r fixed, the candidate fraction does not shrink with the index, so LSH saves a constant factor; with the tuned settings that factor is large enough for pHash LSH to beat a vectorised linear scan at 50,000 references.
Spatial verification (finding 4) accepts true pairs with low Jaccard, which the selective MinHash settings no longer propose, so (b, r) has to be tuned for the verification that follows it, not for raw Jaccard.

## 3. Raw Jaccard is the bottleneck of System B, not retrieval

Visual words find the right image far more often than they can accept it: mean recall@1 0.74 against 0.50 at the 1% threshold.
Examples (exhaustive): flip 0.74 against 0.07; downscale to 50% 0.90 against 0.27; blur sigma 2 0.57 against 0.04.
True matches under these edits have Jaccard similarities inside the distractor range.

## 4. Spatial verification closes most of that gap

`visual-words+ransac` re-ranks visual words' top 10 by the number of ratio-test matches one RANSAC affine map explains.
Recall at the 1% threshold, exhaustive retrieval, visual words without and with RANSAC:

| Condition | Without | With |
| --- | --- | --- |
| JPEG quality 10 | 0.39 | 0.96 |
| Downscale to 50% | 0.27 | 0.92 |
| Blur sigma 2 | 0.04 | 0.66 |
| Crop to 15% of the area | 0.12 | 0.63 |
| Flip | 0.07 | 0.36 |
| crop+jpeg+overlay, level 3 | 0.01 | 0.57 |
| rotate+blur, level 2 | 0.00 | 0.47 |
| Mean over 52 conditions | 0.517 | 0.717 |

The cost: 8.4 ms per query exhaustive and 3.0 ms with LSH (top 10 only), and 2.0 GB of index memory at 50,000 images (about 40 KB per image for descriptors and keypoint positions, against 2.8 KB for visual words alone).
Flips stay weak at the threshold even though RANSAC ranks the source first for 88% of flipped queries: the inlier counts of mirrored matches mostly stay below the threshold of about 25.
Why so few mirrored descriptors survive the ratio test is not yet checked; SIFT descriptors are not mirror-invariant, so symmetric image content is a likely explanation.

## 5. The cascade reaches the best possible combination (RQ4): confirmed

All combinations are calibrated on half of the source images and evaluated on the other half.
Mean recall at the common false-match rate over 52 conditions:

| B system | Retrieval | pHash | B alone | Cascade | Fusion | Either (upper bound) | Settled by pHash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Visual words | exhaustive | 0.480 | 0.502 | 0.722 | 0.722 | 0.727 | 24% |
| Visual words + RANSAC | exhaustive | 0.480 | 0.730 | 0.843 | 0.839 | 0.850 | 20% |
| Visual words + RANSAC | LSH, tuned | 0.480 | 0.610 | 0.782 | 0.782 | 0.785 | 24% |

The cascade is within 0.01 of the best possible combination, and score fusion (each score turned into its tail probability among calibration distractors) is no better, so the cheaper cascade is the one to report.
pHash settles only a fifth of all queries alone at the jointly calibrated thresholds; the saving in SIFT extraction is real but smaller than the MVP's 45% suggested.
Chains that pair a geometric edit with blur or downscaling at level 3, and flips combined with JPEG, remain unsolved by every combination.

## 6. False-match rates are clustered by source image

The realised false-match rate on the held-out half overshoots the 1% target for RANSAC (1.3% to 1.4%).
The reason is that false matches concentrate on a few distractor source images, each transformed 52 times: pHash's 103 false matches at its threshold come from 10 of the 500 distractor images, 38 of them from one.
One distractor (R659377) has 119 RANSAC inliers with an indexed reference without any edit, so DISC21's reference set evidently contains a near-duplicate of it.
Wilson intervals treat the 26,000 distractor queries as independent, which is too optimistic; the false-match rate needs intervals resampled over source images (a cluster bootstrap), and the test split, with 2,000 distractor sources, will be steadier.

## 7. Deduplication: Louvain helps only at high resolution, and the union graph helps most

1,000 planted groups of a source and three copies (distinct edits from `dedup.COPY_CONDITIONS`) in a 10,000-image DISC21 catalogue; best adjusted Rand index per system and method:

| Graph | Best threshold | Connected components | Louvain, resolution 1 | Louvain, resolution 100 |
| --- | --- | --- | --- | --- |
| pHash | d <= 10 | 0.600 | 0.600 | 0.600 |
| Visual words | J >= 0.3 | 0.737 | 0.737 | 0.743 |
| Union of both | d <= 8 or J >= 0.3 | 0.855 | 0.855 | 0.866 |

At strict thresholds the graph has almost no false edges, so connected components are already right and Louvain returns the same partition.
At loose thresholds connected components collapse into one giant component (ARI 0.01 at J >= 0.2), as the plan predicted, but Louvain at the default resolution collapses too (0.07): modularity's resolution limit merges groups of four.
At resolution 100 Louvain recovers most of it (0.74 at J >= 0.2), so its value is robustness to the edge threshold, not a higher peak.
The resolution is a free parameter that needs tuning on a validation catalogue, not on the reported one.
Pair recall stays at 0.79 for the best graph, because some planted copies (flips, chains) reach neither system's threshold.

## 8. A larger vocabulary helps, but moves the s-curve's target (Caltech-101 validation)

Same tuned LSH settings (256 permutations, b, r = 85, 3) at three vocabulary sizes, means over 52 conditions:

| Words | Mean true-pair Jaccard | Recall@1, exhaustive | Recall at threshold, exhaustive | Recall at threshold, LSH | LSH candidates |
| --- | --- | --- | --- | --- | --- |
| 512 | 0.43 | 0.71 | 0.48 | 0.48 | 2,144 (35%) |
| 2,048 | 0.33 | 0.79 | 0.54 | 0.53 | 104 (1.7%) |
| 8,192 | 0.28 | 0.83 | 0.58 | 0.51 | 5 (0.08%) |

A larger vocabulary makes visual words more discriminative, as expected, but lowers every Jaccard similarity, true pairs included, so (b, r) tuned for one vocabulary is too loose for a smaller one and too strict for a larger one.
The vocabulary study (seat 2) and the LSH tuning (seat 3) therefore cannot be done independently: retune (b, r) for each vocabulary size before comparing them.

## 9. Caltech-101 cannot calibrate spatial verification

On Caltech-101, RANSAC raises recall@1 (0.79 to 0.85 at 2,048 words) but not recall at the threshold (0.54 to 0.55), because its calibrated threshold is 81 inliers against about 25 on DISC21.
102 of the 500 Caltech distractor images have more than 25 inliers with some indexed image of their own category.
The two strongest checked by eye (octopus 6 against 5, stegosaurus 38 against 14) are the same artwork with a different colour treatment or compression: real near-duplicates that the split counts as distractors.
Caltech-101 stays useful for development and the notebook demo, but thresholds and false-match rates for verification must come from DISC21.

## 10. Every LSH setting can be scored without searching, and the frozen settings follow from it

`tune` computes each validation query's similarity to all 50,000 indexed images once and predicts recall at the threshold and the candidate fraction for every (b, r).
Checked against the run it starts from: pHash candidate fraction 0.0082 measured against 0.0078 predicted, visual words 0.0225 against 0.0222, and RANSAC recall at the threshold 0.717 against 0.716.

Vocabulary size on DISC21 (exhaustive retrieval, 1% false-match rate):

| Words | Davies-Bouldin | Mean true-pair Jaccard | Recall@1 | Recall at threshold | RANSAC, top 10 |
| --- | --- | --- | --- | --- | --- |
| 512 | 3.02 | 0.45 | 0.66 | 0.47 | 0.66 |
| 2,048 | 2.96 | 0.34 | 0.74 | 0.52 | 0.72 |
| 8,192 | 2.70 | 0.29 | 0.78 | 0.55 | 0.73 |

Each quadrupling of the vocabulary gains less (RANSAC: +0.054, then +0.017), so 8,192 words is frozen; a top-25 shortlist would add 0.003 at 2.5 times the verification cost, so the shortlist stays at 10.

The cheapest settings within 0.005 of exhaustive recall at the threshold: pHash b, r = 12, 16 (0.03% candidates; frozen at 17, 16, which loses 0.002 instead); visual words with 8,192 words b, r = 68, 2 of 256 permutations (2.6%).
For RANSAC no selective setting exists: it accepts true pairs whose Jaccard similarity lies in the range of random pairs, so staying within 0.01 of exhaustive recall needs 35% of the catalogue as candidates, and 4.3% costs 0.05.
The frozen MinHash setting, b, r = 113, 2 of 256 permutations, is shared by all visual-word systems: lossless for visual words alone, 0.05 below exhaustive for RANSAC.
Frozen run flags: `--vocabulary 8192 --num-perm 256 --minhash-bands 113 --minhash-rows 2 --bit-bands 17 --bit-rows 16 --top-k 10`.

## 11. Why each system fails (validation mechanism table)

`failures` gives per condition the median of what each representation keeps of the true pair:

- pHash recall@1 falls below 0.5 once the median Hamming distance of the true pair reaches about 14 bits: crop to 75% (18 bits, recall 0.22), rotation by 5 degrees (18, 0.18), overlay of half the image (18, 0.22), and flips (32, the distance of unrelated images).
- Visual words fail when the query keeps too few keypoints: downscaling to 25% leaves 105 of 300, blur sigma 4 leaves 74, and contrast loss of 90% leaves none, which no local method can recover while pHash still finds all of them.
- RANSAC needs about 25 inliers; under flips the ratio test keeps only 29 matches (300 unedited), since SIFT descriptors are not mirror-invariant, and 21 of them are inliers.
- At the breaking points of RANSAC, 11 of 14 sampled misses are shortlist failures: the true source has more inliers than the wrong answer (6 to 45) but never enters the Jaccard top 10, though most of them would still fall below the threshold of about 25; 2 are contrast-90% queries without keypoints, and 1 (flip+JPEG) is a verification failure, 5 inliers for the source against 6 for a wrong image.

## 12. The frozen settings on validation, before the test split

`run disc21 --split val` with the frozen flags; recall at the 1% threshold with 95% intervals from resampling source images:

| System | Exhaustive | LSH | Candidates | ms per query, exhaustive / LSH |
| --- | --- | --- | --- | --- |
| pHash | 0.480 | 0.506 | 0.04% | 0.047 / 0.018 |
| Visual words | 0.550 | 0.549 | 4.2% | 4.5 / 1.1 |
| Visual words + RANSAC | 0.735 | 0.690 | 4.2% | 9.2 / 3.7 |
| Visual words, containment | 0.153 | 0.284 | 4.2% | 4.5 / 1.1 |

The predictions of finding 10 held: visual words 0.550 predicted, RANSAC 0.733 exhaustive and 0.683 with LSH, candidates 4.3%.
pHash LSH now beats the vectorised scan, by 2.7 times.
pHash LSH scores above exhaustive search because integer distances let the exhaustive threshold use only 0.40% of the 1% budget (the next distance overshoots it), while LSH, which drops some distractors at random, lands at 0.94%.
Containment fails as a score: a reference with many words contains most of any query, so the threshold on distractors rejects nearly every target; it does better with LSH only because LSH removes such references unless their Jaccard similarity is high.

Combinations, held-out half, mean recall at the common false-match rate with 95% intervals:

| B system | Retrieval | pHash | B alone | Cascade | Fusion | Either | Settled by pHash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Visual words | LSH | 0.478 | 0.531 | 0.738 [0.732, 0.743] | 0.738 | 0.741 | 24% |
| Visual words + RANSAC | exhaustive | 0.480 | 0.755 | 0.860 [0.853, 0.867] | 0.853 | 0.866 | 20% |
| Visual words + RANSAC | LSH | 0.478 | 0.705 | 0.831 [0.824, 0.838] | 0.830 | 0.838 | 22% |

The held-out false-match rate of the RANSAC cascade, 1.45%, has the source-bootstrap interval [0.57%, 2.49%], so its overshoot of 1% is within the noise of 250 distractor sources; the Wilson interval on 13,000 distractor queries would have called it significant.

## Open for the seats

- Seat 1: pHash (b, r) beyond one tuned point, and the scale point at 100,000 references to see whether the LSH advantage over the scan grows.
- Seat 2: vocabulary size, the containment score, and whether a larger vocabulary moves Jaccard of true pairs out of the distractor range before RANSAC is needed.
- Seat 3: MinHash (b, r) tuned for the RANSAC stage rather than for raw Jaccard, and the cost of a larger shortlist.
- Seat 4: cluster-bootstrap intervals for false-match rates, the failure analysis of the unsolved chains, and the cascade on the test split once everything is frozen.
- Seat 5: whether RANSAC-verified edges make the deduplication graph clean enough that connected components suffice, and how to choose the Louvain resolution.
