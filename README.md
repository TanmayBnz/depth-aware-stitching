# Depth-Aware Image Stitching

Classical (non-learned) image stitching that uses **sparse depth cues** to weight RANSAC homography estimation, so the background plane is aligned more accurately in scenes with significant parallax — with an automatic "safety net" that falls back to standard RANSAC whenever the depth cues don't actually help.

## Motivation

Standard panorama stitching fits a single global homography to all matched keypoints. This works well for rotation-only or planar scenes, but breaks down under **parallax**: foreground objects at a different depth than the background violate the homography assumption and pull the fit away from the plane you actually care about (usually the background).

This project estimates a **per-keypoint depth weight** from purely geometric cues (no learned depth network) and uses it to bias RANSAC sampling and refinement toward background points — while keeping a fallback to standard RANSAC if the depth-weighted model turns out worse.

## Pipeline

```
Image pair
    │
    ▼
FeatureMatcher            SIFT keypoints + Lowe's ratio test (ratio = 0.75)
    │
    ▼
DepthEstimator             Per-match depth weight in [0, 1] combining:
                              • homography residuals (60%) — primary cue
                              • Sampson distance / epipolar geometry (30%)
                              • spatial coherence via k-NN smoothing (10%)
                            → optional dense depth map via RBF interpolation
    │
    ▼
DepthAwareHomography        1. Standard RANSAC homography (baseline)
                             2. Depth-weighted RANSAC homography
                                (samples biased toward high-weight/background points,
                                 weighted least-squares refinement)
                             3. "Safety net": pick whichever model finds the larger
                                un-weighted inlier consensus set — falls back to
                                standard RANSAC if the depth weights were misleading
    │
    ▼
Warp + evaluate (MAE, background MAE, PSNR against ground-truth overlap)
```

## Repository Structure

```
depth-aware-stitching/
├── src/
│   ├── core/
│   │   ├── feature_matcher.py     # SIFT detection + Lowe's-ratio matching
│   │   ├── depth_estimator.py     # Sparse depth from homography residuals + Sampson distance
│   │   └── homography.py          # Depth-weighted RANSAC with standard-RANSAC safety net
│   ├── download_udis_dataset.py   # Helper for setting up the UDIS-D benchmark
│   ├── test_udis_batch.py         # Batch evaluation over the full UDIS-D test split
│   ├── test1.py                   # Single-pair evaluation with detailed alignment metrics
│   ├── udis_pairs.txt             # List of UDIS-D test image pairs used for evaluation
│   └── results/udis_d/            # Evaluation outputs (metrics JSON + plots) — see below
├── scripts/
│   └── setup_udis.sh              # Creates data dirs and kicks off dataset setup
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/TanmayBnz/depth-aware-stitching.git
cd depth-aware-stitching
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires OpenCV with the `contrib` module (for `SIFT_create`), NumPy/SciPy, and standard scientific-Python tooling. `pymaxflow` (graph-cut seam blending) is optional — see `requirements.txt` for details.

## Usage

### 1. Set up the evaluation dataset

This project evaluates against **[UDIS-D](https://github.com/nie-lang/UnsupervisedDeepImageStitching)** (Unsupervised Deep Image Stitching dataset).

```bash
bash scripts/setup_udis.sh
```

UDIS-D must be downloaded manually (Google Drive link in the linked repo) and extracted to `data/raw/dataset/UDIS-D/`, with `testing/input1/`, `testing/input2/`, and `testing/label/` subfolders.

### 2. Run batch evaluation

```bash
cd src
python test_udis_batch.py \
  --data_dir ../data/raw/dataset/UDIS-D \
  --output_dir results/udis_d \
  --max_pairs 200          # optional cap on number of pairs
```

This runs both standard and depth-weighted stitching on every pair, picks the better model per-pair via the safety net, and writes `detailed_results.json` plus summary plots to the output directory.

### 3. Run a single pair with full diagnostics

```bash
python test1.py
```

Prints keypoint counts, depth-weight statistics, and per-method alignment metrics (MAE, PSNR) for one image pair, useful for debugging or visualizing the depth-weight map.

## Results (UDIS-D test split)

Evaluated on **200 image pairs** from UDIS-D (512×512 resolution); 195 pairs succeeded (5 failed with too few SIFT matches to fit a homography).

| Metric | Standard RANSAC | Depth-weighted RANSAC | Final (safety-net choice) |
|---|---|---|---|
| Mean alignment MAE | 50.62 | 50.21 | **49.46** |
| Mean background-region MAE | 59.35 | 58.34 | **57.62** |
| Mean PSNR (dB) | 37.80 | 37.94 | — |

- **Scene mix:** 165 pairs classified high-parallax, 30 low-parallax (by depth-weight spread).
- **Safety-net behavior:** the depth-weighted model was selected as the better fit on **77 / 195** pairs (~39%); standard RANSAC won on the remaining **118 / 195** (~61%) — i.e. the fallback triggers often, which is by design: it prevents the depth cues from ever making alignment *worse* than the classical baseline.
- On the pairs where depth-weighting was selected, some see substantial background-MAE reductions (the single best pair sees ~48% lower background MAE); on the pairs it isn't selected, the safety net falls back cleanly, so overall performance never regresses below the standard-RANSAC baseline.

Full per-pair metrics are in [`src/results/udis_d/detailed_results.json`](src/results/udis_d/detailed_results.json). Visual comparisons:

- `src/results/udis_d/evaluation_plots.png` / `evaluation_plots_hybrid.png` — aggregate metric distributions
- `src/results/udis_d/best_pair_comparison.png` — side-by-side of the largest-improvement pair

## Notes

- Depth is estimated purely from **2-view geometry** (no monocular depth network), so it's fast and dependency-light, but it degrades on scenes with too few or poorly-distributed matches (see the 5 failed pairs above).
- The "safety net" in `DepthAwareHomography.estimate()` is the key design choice: depth-weighted RANSAC is only trusted when it demonstrably finds a *larger* un-weighted inlier consensus set than standard RANSAC, which bounds the worst-case regression from using (possibly noisy) depth weights.

## Acknowledgements

Evaluated against the [UDIS-D dataset](https://github.com/nie-lang/UnsupervisedDeepImageStitching) from *Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images* (Nie et al.).
