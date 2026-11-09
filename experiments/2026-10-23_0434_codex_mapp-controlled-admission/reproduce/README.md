# Reproduction stages

All work used the shared Python executable recorded in launch.json. No installs or environment overrides were used. The original launch.sh retrieves metadata; the following stage scripts/configurations preserve later prospectively recorded work:

1. processed-assets.sh: original full-stream attempt (both archive transfers truncated and failed validation; do not expect this historical attempt to pass).
2. range-recovery.sh: verified ranges and whole archive checksums; original failed partials retained.
3. archive-inspection.sh: complete inventories and original-table extraction.
4. quantity-audit.sh: amended source arithmetic replay. Initial literal-label failure is separately frozen under attempts/01-literal-experiment-label-mismatch/.
5. independent-verification.sh: independent BSD tar/pandas replay.

Each script has its stage JSON and frozen source under reproduce/source/. Fallback/mirror/public inventory requests were small inline HTTP reads with URLs/purpose/time frozen in their named JSON before retrieval, and their bytes/hashes retained. The fallback dataset ZIP was not downloaded or admitted.

Use a new copy of this family and an empty asset destination when rerunning; the archive and extraction scripts deliberately refuse to overwrite original bytes. Adjust the destination entries consistently in the JSON configurations, retain a record of that path-only change, and execute the corresponding code scripts from the new copy. The exact originally executed commands are the stage shell files. Public metadata can change on later access; results/source-metadata.tar.gz preserves this audit's exact small source inputs, independently checked against source-metadata-manifest.json. Whole archive bytes are reproducible using the recorded public URL, size and SHA1; source-member SHA256s are in archive-inspection.json. Previously cached manuscript provenance is recorded separately. No training/evaluation splits or prediction files exist because this family is a source audit.
