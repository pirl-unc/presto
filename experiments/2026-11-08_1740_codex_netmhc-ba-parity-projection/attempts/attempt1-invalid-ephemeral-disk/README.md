# Attempt 1: invalid ephemeral-disk request

Modal app: `ap-E5gNDZmqElXSoi7Rw8Cp0T`

The submission failed before function creation with:

```text
Function disk request out of bounds: 131072 MiB. Must be between 524288 and
3145728 MiB.
```

No function call ID was created, no archive or current identity file was read,
and no output path was created. The continuation changes only the operational
scratch request from 131072 to 524288 MiB. The source, input hashes, admission,
embargo, target, assay-output, and access contracts are unchanged.
