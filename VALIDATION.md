# Release checks

Checked on October 9, 2026. This records implementation checks, not a new
mathematical proof review or a rerun of the full simulation campaign.

- Nine Python LSD/kernel tests passed.
- The R/C++ integration test passed: the three fast rank matrices agree with
  directly enumerated kernels on an `n=8, p=5` sample, and the spectrum output
  path was also exercised.
- Two theoretical-parameter tests passed.
- Four independence-experiment configuration tests passed.
- An end-to-end smoke run passed for all eight data scenarios at `n=8, p=14`,
  with two replications each, all three kernels, and both powers. It produced
  raw statistics and all three rejection-rate tables. These small-sample
  outputs are not evidence for asymptotic calibration or empirical power.
- Finite centers were compared with the current main-text quadratic formulas
  and supplemental quartic formulas. At `q=0.7`, the expanded code formulas
  for all three quartic means and variances agree with the compact expressions
  in Appendix F to floating-point precision; the tau-star quadratic variance
  agrees as well. This was an algebraic consistency check, not a proof of the
  CLT or a power study.

The copied computational source is unchanged. Local installations, compilation
caches, generated smoke outputs, and manuscript source are excluded from the
release archive. A fresh installation on a second operating system and an
exact reproduction of all published table cells have not been performed.
