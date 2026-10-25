# Intact precursor scale: admission requirement before another kinetic fit

2026-10-25; Codex/GPT-6. Read-only source/methods assessment after ESP source
qualification. This is not a numerical experiment, calibration result or new
predictive evaluation. Original100/120-minute outcomes remain closed.

## Question and evidence reviewed

Could remaining intact precursor independently constrain the global processing
rate that product/precursor ratios leave weakly supported? Existing shared
PXD058121 source and ratio recipe retain the precursor signal, but source admission
already preserves unknown run scaling. Reviewed only original article Methods and
published source reports, not new kinetic values. Source/paragraph hashes are in
`tasks/modular_model/intact-precursor-method-coordinates.json`.

The [primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11874349/) specify
automatic gain control, a maximum accumulation time, whole-acquisition spectrum
averaging and Xtract intensity output. Actual per-scan accumulation/scaling is not
established by the processed-table contract. Literal protocol settings are not
per-acquisition normalization measurements. This does not prove that the supplied
intensity was base-peak normalized or that all run changes are instrumental.

## Why this matters mathematically

For intact substrate, log intensity = log initial amount - cumulative degradation
hazard + run/measurement gain. An unrestricted run gain can absorb any change in
hazard. Supplying its signal alongside same-run product/parent ratios therefore
does not by itself provide an independent quantitative anchor. Fixing every run
gain to one would add an unvalidated assumption, not recover a measured scale.

## Prospective data request and admission rule

Extend existing open Hitlist831, not a duplicate issue. Request source-backed
RAW-to-processed-acquisition links; actual scan accumulation times/AGC behavior;
scan inclusion/averaging and deconvolution intensity definitions/scaling; any
stable independent internal reference and its spike amount/stage; or independently
calibrated residual-substrate measurements with uncertainty. Prefer compact
processed metadata exports to vendor-data reprocessing. Preserve protocol claims
and unknown actual values separately using existing context/calibration APIs.

Do not call the sum of listed deconvoluted intensities TIC or a constant mass
standard. Do not infer a normalizer from the precursor/product signals being fitted.
Do not simply divide intensity by accumulation time without a supported software
intensity definition and validation. A subsequent anchor recipe must independently
replay joins, quantities and normalizer applicability, then preregister fitting and
fresh evaluation. It must remain distinguishable from peptide-specific response
calibration under Hitlist762 and from intracellular/APC concentration measurements.

## Decision and next action

No precursor-decay target admitted now. No model code, loss, parameters, outcomes
or old stopping rules change. The next action is the precise upstream metadata
request and qualification if those records become available. Existing source and
prediction results remain valid within their original scope. Full availability,
short/unobserved products, APC machinery effects and exact/RL biological validation
remain unachieved; this assessment is one candidate constraint, not completion.
