# Registration recovery before execution

The initial registration helper wrote the plan, config and extracted feature
function, then stopped at an AssertionError while matching the formatting of
a monitor template. No stage, model or data preparation was launched. Inspection
confirmed the missing launcher and empty results directory. Recovery created
only the missing monitor/launch shell and task checklist; it changed no data,
feature, split, hyperparameter or resource cap. The explicit empty PYTHONPATH
prevents ambient import paths; all stages still require their frozen inputs.
