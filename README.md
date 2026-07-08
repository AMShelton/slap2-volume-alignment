# ScanImage z-straightening conservative smoothing fix v4

Drop these files into the repository root and restart the Jupyter kernel.

This update makes z-straightening conservative when inter-plane registrations are mostly rejected:

- rejected/non-finite z registrations remain unsupported rather than being extrapolated across the stack;
- only short internal gaps are interpolated;
- long gaps and unsupported end regions remain NaN in the smoothed shift columns;
- `apply_z_straightening` already treats NaN shifts as no-shift, so unsupported planes are not moved by invented offsets;
- smoothing is done within contiguous finite runs, not across unsupported gaps.

This will not magically solve an ill-posed sparse-volume registration problem, but it prevents the pipeline from applying large extrapolated shifts when the QC plot shows that most raw registrations failed.
