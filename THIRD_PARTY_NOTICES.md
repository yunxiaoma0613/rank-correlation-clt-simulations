# Third-party notices

The fast rank-statistic backend uses the R package **independence 1.0.1**,
*Fast Rank-Based Independence Testing*, by Chaim Even-Zohar. Its package
DESCRIPTION declares `GPL (>= 3)`.

- Source distribution: https://cran.r-project.org/src/contrib/Archive/independence/independence_1.0.1.tar.gz
- Package information: https://cran.r-project.org/package=independence
- License text: `licenses/GPL-3.txt`

`experiments/second_order_lsd/src/fast_full_rank_matrix.cpp` includes adapted
rank-counting routines, including a thread-safe adaptation of the Hoeffding
count with the worker-thread R interrupt check removed. Other routines call
symbols from the installed independence shared library. The upstream package
is installed separately rather than bundled as a compiled binary.

The complete code release is distributed under GPL-3.0-or-later with the
authors' permission. Upstream authorship and licensing are retained. See
`LICENSE` for the complete license text.

Rcpp, NumPy, SciPy, Matplotlib, R, and Python are external dependencies and
retain their respective licenses. They are not redistributed in this package.
