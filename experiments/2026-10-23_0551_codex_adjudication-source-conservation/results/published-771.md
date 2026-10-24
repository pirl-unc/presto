## Verified combined source generation — October 23, 2026

[Full report and immutable reproduction](https://github.com/pirl-unc/presto/blob/82feaf2/experiments/2026-10-23_0551_codex_adjudication-source-conservation/README.md). Executed Hitlist `0219827`
merges main 2.0.4 without dropping its nine multimer reviews. New external
successors use schema9/catalog12 and accept verified reference or adjudication
predecessors; both historical schema8 families remain readable, and ambiguous
envelopes are rejected. This addresses the integration defect in #771.

The actual combined package retains all126 predecessor files, all33 MAPP and
23 calibration files, and all9 review claims. Independent hashing/direct Parquet
replay checks183files and1,853,840mixed-grain rows across32canonical query
relations. Allfour existing recipe payloads, contributors and partitions remain
byte-identical. Source provenance changes to the new generation. The initial
standalone/canonical recipe-envelope baseline error and amended verification
are retained; no producer or target was changed/rerun to obtain a pass.

Normal local `./test.sh` passes3,317tests/10warnings in321.51s with the unchanged
memoryguard andoneworker. Format/lint/70focused checks/strict docs pass. Producer
148.01s/3.28GiBpeakRSS; independent verifier12.83s/0.64GiB, local CPU only.
Code is on draft767; 2.0.5 remains unreleased. HostedCI/review/merge/release are
separate gates. Controls admission remains in #759; no new biological fit or
concentration claim is made here.
