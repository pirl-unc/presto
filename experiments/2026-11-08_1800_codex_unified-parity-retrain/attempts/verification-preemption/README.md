# Independent verifier preemption and raw comparison stop

Call `fc-01M6YTYJTA4Z33FBNACMH886HY` was preempted after writing reference
admission and the parity manifest. A platform restart failed closed because the
partial immutable output directory already existed. It contains no receipt and
is not evidence of verification. It remains preserved at
`preflight-v2-verification/` on the Modal volume.

The retry-safe continuation writes locally and atomically publishes only a
complete tree. Call `fc-01M6YWM3TV4JF1CT6HACAQW0M7` completed both condition
passes but its live-Python versus JSON-loaded equality check stopped at
`reference/model` due tuple/list representation. Its full persisted receipt is
26,182 bytes, SHA-256
`fdce01321abedf12f0e3aa961f0bd8ab93bade1896a101bc65530d6015178326`.

The hash-pinned canonical-JSON closure independently compared every declared
scientific field and passed. Neither source receipt was rewritten.
