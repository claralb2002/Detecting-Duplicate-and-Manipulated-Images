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

## 13. The aligned pipeline: edits at full resolution, same frozen settings

The pipeline on `main` applies each edit to the full-resolution image and only then makes the 512-pixel copy for SIFT, as a re-upload edits the original; before, edits were applied to the 512-pixel copy.
DISC21 references are about 1,024 pixels on the longer side, so a downscale to 50% now leaves SIFT nearly the same image, and the strengths of findings 1-12 are not comparable one to one with this section.
Everything else is unchanged: the vocabulary is fitted on the 38,013-image fitting pool, and the run flags are the frozen ones of finding 10.

Recall at the 1% threshold on validation, 95% intervals from resampling source images:

| System | Exhaustive | LSH | Candidates | ms per query, exhaustive / LSH |
| --- | --- | --- | --- | --- |
| pHash | 0.481 [0.478, 0.484] | 0.508 [0.504, 0.511] | 0.04% | 0.048 / 0.017 |
| Visual words | 0.610 [0.604, 0.616] | 0.609 [0.603, 0.615] | 4.5% | 4.4 / 1.2 |
| Visual words + RANSAC | 0.837 [0.827, 0.845] | 0.793 [0.785, 0.800] | 4.5% | 12.1 / 6.7 |
| Visual words, containment | 0.202 [0.199, 0.205] | 0.408 [0.401, 0.415] | 4.5% | 4.3 / 1.2 |

Visual words gain 0.06 and RANSAC 0.10 over finding 12; pHash is unchanged, since it reduces its input to 32 x 32 pixels whether that input has 512 pixels on the longer side or the full resolution.The s-curve still predicts LSH recall@1: pHash 0.543 measured against 0.540 predicted, visual words 0.789 against 0.783.

`tune` on this run confirms the frozen settings rather than moving them:

- Its predictions match the run: candidates 0.0452 measured against 0.0453 predicted for visual words, RANSAC recall at the threshold 0.837 against 0.837.
- 8,192 words beat 2,048 again (recall at the threshold 0.610 against 0.579; RANSAC, top 10, 0.837 against 0.806).
- The frozen MinHash setting is lossless for visual words alone (0.609 against 0.610 exhaustive).
- For RANSAC the gap between rows of 2 and rows of 1 remains: with 2 rows no setting predicts more than about 0.77, and staying within 0.005 of exhaustive needs 1 row and 48% of the catalogue as candidates.
- A top-25 shortlist would add 0.007 at 2.5 times the verification cost, so the shortlist stays at 10.

`tune` predicts RANSAC with LSH at 0.767 at the frozen setting, but the run measures 0.793.
`tune` keeps the exhaustive threshold, while `run` calibrates the LSH threshold on LSH's own distractor scores; LSH removes some high-scoring distractors, so its threshold is lower and admits more targets, as for pHash in finding 12.
The prediction is therefore a lower bound for the LSH systems.

Combinations, held-out half, mean recall at the common false-match rate with 95% intervals:

| B system | Retrieval | pHash | B alone | Cascade | Fusion | Either | Settled by pHash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Visual words | LSH | 0.508 | 0.584 | 0.730 [0.723, 0.737] | 0.726 | 0.741 | 22% |
| Visual words + RANSAC | exhaustive | 0.481 | 0.867 | 0.915 [0.906, 0.923] | 0.891 | 0.920 | 21% |
| Visual words + RANSAC | LSH | 0.508 | 0.800 | 0.877 [0.869, 0.885] | 0.867 | 0.883 | 21% |

The cascade stays within 0.006 of the best possible combination and beats fusion, as in finding 5.
The held-out false-match rate of the exhaustive RANSAC cascade is 1.44%, with the source-bootstrap interval [0.45%, 2.63%], so it again sits within the noise of 250 distractor sources.

## 14. Test split: validation carries over

`run disc21 --split test` with the frozen flags: 2,000 targets and 2,000 distractors under 52 conditions, 208,000 queries, thresholds from validation unchanged.

| System | Exhaustive | LSH | False-match rate, exhaustive / LSH |
| --- | --- | --- | --- |
| pHash | 0.482 [0.480, 0.484] | 0.510 [0.508, 0.512] | 0.15% / 0.61% |
| Visual words | 0.610 [0.606, 0.613] | 0.609 [0.606, 0.612] | 1.24% / 1.24% |
| Visual words + RANSAC | 0.832 [0.828, 0.836] | 0.790 [0.786, 0.794] | 0.40% / 0.58% |
| Visual words, containment | 0.203 [0.202, 0.205] | 0.413 [0.410, 0.417] | 0.91% / 1.32% |

Every recall is within 0.006 of validation (finding 13), so the frozen settings did not overfit validation.
The thresholds transfer less evenly: visual words overshoot the 1% target (1.24% [0.93%, 1.59%]), while RANSAC's inlier threshold, set on 250 validation distractor sources, is conservative on test (0.40%).
The s-curve again predicts LSH recall@1: pHash 0.546 measured against 0.542 predicted, visual words 0.784 against 0.779, and candidate rates within 0.002.

Combinations, calibrated on half of the test source images and evaluated on the other half:

| B system | Retrieval | pHash | B alone | Cascade | Fusion | Either | Settled by pHash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Visual words | LSH | 0.509 | 0.598 | 0.739 [0.736, 0.742] | 0.734 | 0.753 | 24% |
| Visual words + RANSAC | exhaustive | 0.521 | 0.868 | 0.919 [0.915, 0.922] | 0.912 | 0.925 | 22% |
| Visual words + RANSAC | LSH | 0.509 | 0.793 | 0.876 [0.872, 0.879] | 0.872 | 0.880 | 24% |

With RANSAC the cascade stays within 0.006 of the best possible combination, with visual words alone within 0.014; the held-out false-match rate of the RANSAC cascade is 0.53% exhaustive and 0.56% with LSH.
Unlike the single systems, the combinations are calibrated within the test split; the halves share no source image.

Failure table (`failures`, medians over targets), where each representation breaks:

- pHash: recall@1 falls below 0.5 as the median Hamming distance of the true pair grows from 12 to 18 bits; at 18 bits (crop to 75%, rotation by 5 degrees, overlay of half the image) it is about 0.2, and flips sit at 32 bits, the distance of unrelated images.
- Visual words: when blur, downscaling or contrast loss remove keypoints (blur sigma 8 leaves 69 of 300, downscale to 10% 64, contrast loss of 90% none), or when an edit changes the descriptors themselves: doubled brightness keeps all 300 keypoints, but the true pair's median Jaccard falls to 0.055 and recall@1 to 0.29.
- RANSAC: flips keep a median of 18 ratio-test matches and 11 inliers, against 235 and 234 for an unedited query; flip+JPEG at level 3 keeps 6 inliers.
- The gallery of 69 sampled misses is written locally under `data/failures/` and never committed.

## 15. Meta's development queries and deduplication on the aligned pipeline

`run disc21 --split dev` with the frozen flags: thresholds calibrated on `dev_calibration`, scores on `dev_report` (5,000 queries with a match in the index, 5,000 without).

| System | Recall at 1%, exhaustive | Recall at 1%, LSH | Micro-AP, exhaustive | Micro-AP, LSH |
| --- | --- | --- | --- | --- |
| pHash | 0.192 [0.182, 0.203] | 0.184 | 0.228 | 0.202 |
| Visual words | 0.153 [0.143, 0.163] | 0.151 | 0.291 | 0.256 |
| Visual words + RANSAC | 0.471 [0.458, 0.485] | 0.361 | 0.474 | 0.363 |
| Visual words, containment | 0.042 | 0.077 | 0.223 | 0.213 |

Meta's queries are much harder than the generated ones: exhaustive RANSAC's recall at the threshold is 0.471, against 0.832 on the test split.
The RANSAC cascade reaches 0.510 [0.496, 0.523] exhaustive and 0.420 with LSH; fusion and the best possible combination are within 0.004, and pHash settles only 10% of queries, against 22% on the test split.
The s-curve still predicts LSH recall@1 for visual words, 0.335 measured against 0.331 predicted.

Douze et al. (2021, Table 2) report micro-AP on the development queries against all 1,000,000 references: GIST 14.4%, MultiGrain 16.5% (36.4% with score normalisation) and HOW+ASMK 17.3% (37.2%).
RANSAC's 47.4% is not comparable one to one: our index holds 50,000 references, so far fewer can be confused with the source, and `dev_report` has a match for half of its queries against a fifth in Meta's set.
Both make our score optimistic, so the comparison suggests that a classical local-feature pipeline with spatial verification is competitive with the published baselines, not that it beats them.

Deduplication: `dedup disc21 --index-size 10000 --groups 1000 --copies 3` with the frozen LSH flags, 13,000 images, best adjusted Rand index per graph:

| Graph | Threshold | Connected components | Louvain, resolution 100 |
| --- | --- | --- | --- |
| pHash | d <= 10 | 0.554 | 0.554 |
| Visual words | J >= 0.2 | 0.813 | 0.840 |
| Visual words | J >= 0.15 | 0.429 | 0.853 |
| Visual words | J >= 0.1 | 0.008 | 0.708 |
| RANSAC-verified | inliers >= 15 | 0.903 | 0.903 |
| RANSAC-verified or pHash | inliers >= 15 or d <= 8 | 0.903 | 0.906 |

RANSAC-verified edges answer the seat-5 question: they are clean enough (pair precision 0.997) that connected components suffice, and Louvain changes nothing.
On unverified visual-word graphs Louvain at resolution 100 is what keeps loose thresholds usable (0.708 against 0.008 at J >= 0.1), so its value remains robustness to the edge threshold.
The unverified union graph peaks at 0.777, below visual words alone, because the sweep joins the loose Jaccard thresholds where visual words peak with pHash distances of 12 to 14 bits, where pHash's own edges are already noisy (pair precision 0.83 at d <= 12, 0.40 at d <= 14).
At a fixed Jaccard threshold a strict pHash distance does help: 0.594 for J >= 0.3 alone, 0.696 joined with d <= 8.
Pair recall stays at most 0.83, since flips and level-3 chains reach no threshold.
These thresholds and resolutions are chosen on this catalogue; `dedup --seed 1` plants a new catalogue on which the chosen settings are to be reported.

## 16. Scalability from 10,000 to 100,000 references (validation queries)

`run disc21 --split val` with the frozen flags at `--index-size 10000` and `25000`, the full index (50,000) and `--scale` (100,000); each point calibrates its own thresholds, and the vocabulary is the same at every point.

Milliseconds per query (search after feature extraction):

| System | Mode | 10,000 | 25,000 | 50,000 | 100,000 |
| --- | --- | --- | --- | --- | --- |
| pHash | exhaustive | 0.017 | 0.035 | 0.048 | 0.106 |
| pHash | LSH | 0.021 | 0.023 | 0.017 | 0.023 |
| Visual words | exhaustive | 1.05 | 2.89 | 4.38 | 10.13 |
| Visual words | LSH | 0.21 | 0.80 | 1.16 | 2.65 |
| Visual words + RANSAC | exhaustive | 7.5 | 10.5 | 12.1 | 21.1 |
| Visual words + RANSAC | LSH | 6.1 | 7.0 | 6.7 | 9.0 |

Recall at the 1% threshold:

| System | Mode | 10,000 | 25,000 | 50,000 | 100,000 |
| --- | --- | --- | --- | --- | --- |
| pHash | exhaustive | 0.519 | 0.519 | 0.481 | 0.481 |
| pHash | LSH | 0.528 | 0.508 | 0.508 | 0.478 |
| Visual words | exhaustive | 0.655 | 0.630 | 0.610 | 0.585 |
| Visual words | LSH | 0.653 | 0.628 | 0.609 | 0.585 |
| Visual words + RANSAC | exhaustive | 0.894 | 0.866 | 0.837 | 0.804 |
| Visual words + RANSAC | LSH | 0.803 | 0.798 | 0.793 | 0.772 |

With r fixed, LSH proposes a constant fraction of the index (0.04% for pHash, 4.5% for visual words at every size), so its cost grows linearly too, only by a constant factor less: visual words with LSH are 3.6 to 5.1 times faster than the scan at every size.
pHash's candidates are so few (43 at 100,000) that the lookup itself dominates, so its LSH time stays near 0.02 ms while the scan grows sixfold; LSH is no faster than the scan at 10,000 and 4.6 times faster at 100,000.
RANSAC's cost is dominated by verifying its 10 shortlisted images, so exhaustive retrieval adds only the Jaccard scan in front of it.
Index memory grows linearly: 21 MB for pHash, 349 MB for visual words, and 4.1 GB for RANSAC at 100,000 images, about 41 KB per image for descriptors and keypoint positions.
Recall at the threshold falls as the index grows, by 0.09 for exhaustive RANSAC from 10,000 to 100,000, because more references compete with the source and the threshold set on distractors rises (from 6.1 to 18.1 inliers for exhaustive RANSAC, from 0.154 to 0.197 Jaccard for visual words); the LSH variant loses less (0.03), since LSH also hides some of those competitors.

## 17. Why RANSAC misses: verification shortfall, retrieval, ranking (test split)

Each target of `visual-words+ransac` on the test split (104,000 per mode) falls into one class, checked in this order at that mode's own operating point (inlier threshold 12.2 exhaustive, 7.1 with LSH, both set on validation):

1. found: the source is the top answer and scores above the threshold;
2. too few inliers: the true pair itself does not exceed the threshold, so no retrieval could have accepted it;
3. not proposed by LSH: the true pair would pass, but MinHash never made the source a candidate;
4. shortlist or outranked: the true pair would pass and was retrievable, but it did not enter the top 10 by Jaccard similarity or a wrong image scored higher.

| Mode | Found | Too few inliers | Not proposed by LSH | Shortlist or outranked |
| --- | --- | --- | --- | --- |
| Exhaustive | 0.832 | 0.137 | - | 0.031 |
| LSH | 0.790 | 0.083 | 0.119 | 0.009 |

"Too few inliers" is a failure to meet the geometric threshold, mostly because too few useful features survive the edit, not a pure representation failure: contrast loss of 90% (no keypoints, 1.00), flips with JPEG (0.70 to 0.92, since SIFT descriptors are not mirror-invariant), blur sigma 8 (0.60) and flips (0.59).
The two rows are not comparable class by class, because the LSH threshold is lower: LSH accepts some true pairs with 7 to 12 inliers that the exhaustive threshold rejects, so its "too few inliers" share is smaller.
With LSH the largest class of misses is retrieval: MinHash does not propose sources whose Jaccard similarity has collapsed although enough inliers survive, above all under strong blur and downscaling (rotate+blur level 3, 0.56; blur sigma 8, 0.49; doubled brightness, 0.47; downscale to 10%, 0.46).
Ranking errors are rare in both modes (3.1% and 0.9%), so a longer shortlist would gain little, as `tune` predicted (finding 13).

## 18. One protocol for the test combinations, and near-duplicate distractors

**Combinations calibrated on validation.**
Finding 14 calibrated the combinations on half of the test sources, while the single systems reuse their validation thresholds, so the same system received two numbers (RANSAC exhaustive 0.832 alone, 0.868 in the combination table).
`analyze` now calibrates a test file's combinations on the validation file of the same run and evaluates them on the whole test split (`cascade.compare_calibrated_on`); this table supersedes the one in finding 14:

| B system | Retrieval | pHash | B alone | Cascade | Fusion | Either | False matches, cascade | Settled by pHash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Visual words + RANSAC | exhaustive | 0.482 | 0.832 | 0.887 [0.884, 0.890] | 0.870 | 0.893 | 0.38% | 20% |
| Visual words + RANSAC | LSH | 0.510 | 0.790 | 0.867 [0.864, 0.870] | 0.850 | 0.877 | 0.55% | 20% |
| Visual words | LSH | 0.510 | 0.609 | 0.748 [0.745, 0.750] | 0.737 | 0.760 | 1.24% | 22% |

The RANSAC cascade gains 0.055 [0.054, 0.057] exhaustive and 0.077 [0.076, 0.079] with LSH over RANSAC alone, stays within 0.011 of "either", and now beats fusion by 0.017, which it did not under the half-split protocol.
pHash settles 41% of the targets and none of the distractors, so the share of queries that skip SIFT depends on the share of copies among the queries.
The development queries keep their own protocol (calibrated on `dev_calibration`), where fusion is 0.002 and 0.001 above the cascade (finding 15).

**Near-duplicates among the distractors.**
False matches concentrate on a few distractor sources: on validation, RANSAC's 260 come from 15 of 500 sources, 88% of them from sources accepted under 20 or more of the 52 conditions; on test, 415 come from 31 of 2,000.
9 validation and 14 test sources are accepted by RANSAC without any edit or at JPEG quality 90.
The two worst validation sources, viewed side by side with the indexed image they match, are consecutive frames of the same video, so these "false matches" are genuine near-duplicates that DISC21's reference set contains and our split counts as distractors (finding 6 found the first of them).
They set the thresholds: dropping the 18 validation sources whose unedited version any system accepts lowers RANSAC's exhaustive threshold from 12.2 to 0.19, below a single inlier (the score is inliers plus Jaccard similarity), so even a top answer without an accepted homography passes on its Jaccard similarity; on test without its 48 such sources RANSAC's recall at the threshold rises from 0.832 to 0.877 at a false-match rate of 1.22% instead of 0.40%.
The reported recall is therefore conservative for RANSAC, and its intervals are conditional on thresholds that a few mislabelled sources determine; the rising thresholds with index size (finding 16) may partly come from more such near-duplicates entering the index, which is not separated here.

**Vocabulary diagnostic.** The `tune` run of finding 13 printed Davies-Bouldin indices of 2.886 for 2,048 words and 2.525 for 8,192 (lower is more compact), from the vocabulary metadata in `data/cache/`.

## Open for the seats

- Seat 1: pHash (b, r) beyond one tuned point, and the scale point at 100,000 references to see whether the LSH advantage over the scan grows.
- Seat 2: vocabulary size, the containment score, and whether a larger vocabulary moves Jaccard of true pairs out of the distractor range before RANSAC is needed.
- Seat 3: MinHash (b, r) tuned for the RANSAC stage rather than for raw Jaccard, and the cost of a larger shortlist.
- Seat 4: cluster-bootstrap intervals for false-match rates, the failure analysis of the unsolved chains, and the cascade on the test split once everything is frozen.
- Seat 5: whether RANSAC-verified edges make the deduplication graph clean enough that connected components suffice, and how to choose the Louvain resolution.
