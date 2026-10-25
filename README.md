# k-Nearest Neighbours for Data Streams

Online, single-pass kNN classifiers for evolving data streams, built on [CapyMOA](https://capymoa.org/) / MOA. University assignment for COMPX523 (Machine Learning for Data Streams). The interesting part isn't kNN itself but how the **windowing strategy** changes behaviour under class imbalance and concept drift.

## Three windowing strategies

All three are incremental classifiers that test-then-train on each instance with a bounded memory window:

| Variant | File class | Idea |
|---|---|---|
| **Base kNN** | `KNNBase` | One global FIFO sliding window of the last `w` instances. |
| **Class-weighted (cw)** | `KNNClassWindows` | A **separate window per class**, so minority classes keep their own recent history instead of being flushed out by a dominant class. |
| **Age-aware (asw)** | `KNNAgeWeighted` | A single window where, on overflow, an instance is **probabilistically ejected with weight ∝ 1.2^age** (log-sum-exp for numerical stability). Old instances leave first, so the window adapts to drift while keeping a soft recency bias. |

## Results

Tested on the first 50,000 instances of the `electricity` stream and of a synthetic `RBF` stream with concept drift, across window sizes {100, 500} and k {3, 21}. Single runs per setting. Accuracy (%):

| Variant | electricity (best) | RBF (best) |
|---|---|---|
| Base kNN | 86.2 (w=100, k=3) | 87.4 (w=500, k=3) |
| **Class-weighted** | **87.0** (w=100, k=3) | **87.5** (w=500, k=3) |
| Age-aware | 86.1 (w=100, k=3) | 87.4 (w=500, k=3) |

Key finding: the **class-weighted** variant held up best in the small-window RBF case; at the best settings all three are within a point of each other. On the imbalanced/drifting RBF stream, plain kNN with a small window collapsed to ~66%, while per-class windows held at ~85%, because each class retained enough neighbours to vote with. Larger k (21) consistently hurt on these streams, because the local decision boundary is better served by a tight neighbourhood. Full numbers are in `results.csv`.

## Repository layout

```
knnAssignment.py      The three kNN stream classifiers (Base, cw, asw)
run_experiments.py    Sweeps datasets × window sizes × k and writes results.csv
experiments.ipynb     Exploratory notebook (outputs cleared)
results.csv           Full metrics: accuracy, recall, precision, wall-clock
```

> **Data not included.** The `electricity` and `RBF` streams are MOA datasets; CapyMOA downloads them on the first run of `run_experiments.py` (Java 17 needed). The age-aware variant draws its ejections from a fixed seed, so its numbers repeat exactly; the other two variants are deterministic.

## Tech

Python · CapyMOA / MOA (JPype) · single-pass streaming evaluation.

## License

MIT (see `LICENSE`). The MOA datasets are not included and are not covered by the licence.

---
*University coursework (COMPX523). Shared as a portfolio sample.*
