# Frozen profiled-offset fitting receipt

Qualification passed before any fitting. Same two physical models and original
200 draws/Adam settings/source features/targets as the joint-offset family.
Allocate 78 common coordinates and one raw eta only in the fragment arm.
Compute one offset from every admitted observation of the selected condition at
current parameters. Detach it in the block loss. Reuse only selected-block values
from that same forward state; check all four condition times numerically.

Resource cap is 8400 seconds per arm and 4 GiB peak RSS.
The cap was frozen before fitting using the longest-parent target-free cost
26.086 s times 200 updates with 50% margin plus 300 s evaluation,
rounded up to 600 s. There is no outcome-driven budget extension. All required
queries keep the 2/4-step positive/log-agreement guard and selected post-update
check. Stop on any failure; no floors, dropped rows, eta reset or checkpoint rescue.
Final checkpoint only; original test parents excluded. Preserve all states,
source-linked train/development predictions, results and failures.
