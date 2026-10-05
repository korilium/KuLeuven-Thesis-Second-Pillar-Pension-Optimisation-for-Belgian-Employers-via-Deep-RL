# Stale: produced before the REVIEW fix round

These rate-mode figures were produced under the old discounted objective and the
legacy conventions (model years starting at the last observation, no drift
correction, level-dispersion CE moments). They are not results of the canonical
configuration. Regenerate them with `--rates=vasicek` (robustness: `--rates=hull_white_p`)
once `LONG_RATE_P` is set (docs/REVIEW.md, M7 and M9). `hull_white` (risk-neutral)
cannot be scored under the retirement-date objective.
