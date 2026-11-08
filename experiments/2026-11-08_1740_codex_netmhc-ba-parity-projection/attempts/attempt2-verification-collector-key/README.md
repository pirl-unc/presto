# Attempt 2: verification collector key mismatch

Verification app `ap-skFXSxLmvLle9aU67Rw5uZ` and call
`fc-01M6YNS41PTJMGR2NFJV8SASVR` completed and returned a receipt. The local
collector then raised `KeyError: 'summary_bytes'` because it treated the
verification receipt's `summary_path` as a projection receipt.

The collection continuation changes only the branch predicate from
`summary_path` to `summary_bytes`. It reuses the completed verification call and
does not rerun projection or verification. No remote artifact or scientific
contract changes.
