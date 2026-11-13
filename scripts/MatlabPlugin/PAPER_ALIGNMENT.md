# Paper and pipeline alignment

Current defaults follow the published Windowed TEP notebook, including the
quadratic pre fit, free decay, inclusive onset/peak masks and channel screening.
See [TEP alignment, source provenance and validation](../TEP_ALIGNMENT.md).

The plugin supports Gaussian and Fourier-resampled gap filling; demos disable
it explicitly. See [FILLING.md](../FILLING.md).

[Historical figure alignment](../PAPER_ALIGNMENT.md) and the
[controlled numerical-branch comparison](../PAPER_PLUGIN_COMPARISON.md) describe
the explicit figure profile, not the current TEP defaults. Original Lagrange
monomial residuals and different native Gaussian samples remain documented
numerical differences.
