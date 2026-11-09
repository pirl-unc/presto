# Primary receipt return-transport failure

Primary call `fc-01M6YSQBZ1AKY6WW2NN1ZBQD9N` completed reference and parity,
wrote and committed `preflight-v2-primary/receipt.json`, then failed while
returning the complete Python receipt. Modal selected external blob transport
for the return object; the worker intentionally had `block_network=True`, so DNS
resolution for the blob endpoint failed.

The committed receipt was recovered without rerunning the primary pass. It is
25,681 bytes, SHA-256
`6efe6b3f64cbff080f9b03b9dce0ca1906a008d487fee3bd7d9eb8532ac6aaa2`,
and reports both conditions passed. Later functions return compact artifact
pointers while full receipts remain on the volume.
