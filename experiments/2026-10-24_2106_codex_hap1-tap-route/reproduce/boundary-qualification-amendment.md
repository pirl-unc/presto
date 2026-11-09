# Post-fit qualification coverage correction

The independent prediction/gradient replay passed. A closure review then found
that qualify.py's epsilon=1 and monotonicity checks evaluated the independent
algebra, rather than calling the production log_response function at that boundary.
The finite-grid production forward/gradient checks did run before all fits, but
the named boundary gate's implementation coverage was incomplete at launch.

Preserve original qualifier, log, fits, source hashes, gates and outcomes. Add a
separate checker that calls the unchanged production function with eta=+infinity
(epsilon exactly1), compares finite increasing eta pairs, and verifies exact-forward
equality for the sampled-gradient implementation. Original forward tolerance1e-12.
Do not refit or alter the model. Record this as post-fit qualification, never as a
prospective gate that was fully executed. If it fails, mark the family unqualified
and stop interpretation until a separately registered correction is resolved.

Also illustrate the endpoint's algebraic limitation: multiplying baseline and KO
amounts by the same positive peptide-specific quantity leaves their ratio unchanged.
This is a mathematical identifiability example, not a new biological experiment.
