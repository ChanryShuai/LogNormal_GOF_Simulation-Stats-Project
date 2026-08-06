# Critical Values for the KS, CvM, and AD Goodness-of-Fit Tests for the Log-Normal Distribution with Estimated Parameters

**Author:** Chanry Shuai

## Overview

Standard critical values for empirical distribution function (EDF) goodness-of-fit tests — Kolmogorov–Smirnov (KS), Cramér–von Mises (CvM), and Anderson–Darling (AD) — are invalid when distribution parameters are estimated from the sample rather than fully specified. This project uses Monte Carlo simulation to generate finite-sample critical values for these three tests under a log-normal null hypothesis with estimated parameters, and validates the results against established benchmarks for the normal distribution.

## Repository Contents

| File | Description |
|---|---|
| `LogNormal_KS_CVM_AD_v2.pdf` | Revised report (v2) with corrected theoretical framing and validation discussion. |
| `shuai_lognormal_ks_cvm_ad.py` | Python script implementing the Monte Carlo simulation and generating the critical value tables used in the report. |

## Method Summary

For sample sizes n = {10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000}, the script:

1. Draws a random sample from a standard log-normal distribution.
2. Estimates parameters on the log scale (sample mean and sample standard deviation, `ddof=1`).
3. Transforms the data to `z` values via the fitted CDF.
4. Computes the KS, CvM, and AD statistics.
5. Repeats 10,000 times per sample size to build an empirical null distribution.
6. Extracts the 0.75, 0.90, 0.95, and 0.99 quantiles as critical values.

## Requirements

- Python 3
- NumPy
- pandas
- SciPy

## Running the Simulation

```bash
python shuai_lognormal_ks_cvm_ad.py
```

This prints the critical value tables for the KS, CvM, and AD tests and reports total execution time.

## Key Finding

Because the logarithm of a log-normal random variable is normally distributed, fitting a log-normal distribution to the data is equivalent to fitting a normal distribution to the log-transformed data. As a result, the KS, CvM, and AD statistics for the log-normal case are theoretically identical to their normal-distribution counterparts under matching parameter estimation conventions. The simulation numerically confirms this equivalence, and the results show that standard normal estimated-parameter critical value tables can be applied directly to log-normal goodness-of-fit testing.
