# Bootstrap amendment before metadata retry

The first invocation failed before the resource guard initialized: code/inspect.py
shadowed stdlib inspect during importlib.metadata import. No source query ran and
no model input was interpreted. The original launcher, request and stage source
remain byte-for-byte frozen; the failure has no reliable resource peak receipt.

Use the standard PYTHONSAFEPATH=1 startup option in a separately frozen wrapper.
It prevents automatic script-directory insertion for both launcher and its child,
while preserving the explicitly supplied Hitlist PYTHONPATH. No environment install,
source rewrite, numerical/data/eligibility change or resource relaxation occurs.
The new request pins this wrapper and records the inherited bootstrap environment.
Retry is inspect attempt2; preserve all original attempt1 artifacts.
