# k-Nearest Neighbours for Data Streams

Online, single-pass kNN classifiers for evolving data streams, built on [CapyMOA](https://capymoa.org/) / MOA. University assignment for COMPX523 (Machine Learning for Data Streams). The interesting part isn't kNN itself — it's how the **windowing strategy** changes behaviour under class imbalance and concept drift.

## Three windowing strategies

All three are incremental classifiers that test-then-train on each instance with a bounded memory window:

| Variant | File class | Idea |
|---|---|---|
| **Base kNN** | `KNNBase` | One global FIFO sliding window of the last `w` instances. |
| **Class-weighted (cw)** | `KNN_cw_1689664` | A **separate window per class**, so minority classes keep their own recent history instead of being flushed out by a dominant class. |
| **Age-aware (asw)** | `KNN_asw_1689664` | A single window where, on overflow, an instance is **probabilistically ejected with weight ∝ 1.2^age** (log-sum-exp for numerical stability) — old instances leave first, adapting to drift while keeping a soft recency bias. |

## Results

Tested on the `electricity` stream and a synthetic `RBF` stream with concept drift, across window sizes {100, 500} and k {3, 21}. Accuracy (%):

| Variant | electricity (best) | RBF (best) |
|---|---|---|
| Base kNN | 86.2 (w=100, k=3) | 87.4 (w=500, k=3) |
| **Class-weighted** | **87.0** (w=100, k=3) | **87.5** (w=500, k=3) |
| Age-aware | 86.1 (w=100, k=3) | 87.4 (w=500, k=3) |

Key finding: the **class-weighted** variant is the most robust. On the imbalanced/drifting RBF stream, plain kNN with a small window collapsed to ~66%, while per-class windows held at ~85%, because each class retained enough neighbours to vote with. Larger k (21) consistently hurt on these streams — the local decision boundary is better served by a tight neighbourhood. Full numbers in `results.csv`; methodology and discussion in `REPORT.pdf`.

## Repository layout

```
knnAssignment.py      The three kNN stream classifiers (Base, cw, asw)
assignment1.py        Final submitted experiment driver
run_experiments.py    Sweeps datasets × window sizes × k and writes results.csv
experiments.ipynb     Exploratory notebook (outputs cleared)
generate_report.py    Programmatic PDF report generator (ReportLab)
results.csv           Full metrics: accuracy, recall, precision, wall-clock
REPORT.pdf            Written report
```

> **Data not included.** The `electricity` and `RBF` ARFF streams are MOA-provided datasets; point the scripts at your own MOA data directory to reproduce.

## Tech

Python · CapyMOA / MOA (JPype) · single-pass streaming evaluation.

---
*University coursework (COMPX523). Shared as a portfolio sample.*
