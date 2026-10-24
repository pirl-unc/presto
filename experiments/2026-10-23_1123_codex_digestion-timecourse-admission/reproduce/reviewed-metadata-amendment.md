# Reviewed source compatibility correction

The first materialization succeeded but its comparison stopped on contexts.
No model fit or temporal response metric ran. Original script/config/log and
both immutable recipe exports remain. Inspection shows the contexts embed the
source contract as fields_json; the reviewed contract adds exactly nine expected
inventory counts. Every other original source column/value is identical.

The new verification-only script permits exactly those nine additive census
keys, preserves all previous expected values, and requires every other context
field and every source quantitative/mapping/replicate value to match. Derived
recipe changes must be only the corresponding embedded context JSON and the
already declared explicit units, quantity-scale and slot status annotations.
No rematerialization, curation change, target change or resplit is authorized.
