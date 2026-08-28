# RIDE Remote Runtime Sidecar

This subproject is built and deployed independently from the `vllm` package. It
terminates TLS 1.3 with mandatory client certificates, matches the exact client
certificate SHA-256 fingerprint against an operator allowlist, and transparently
forwards one bounded `VLCB1` request to the core-owned local Unix socket.

It does not parse, create, modify, or authorize control actions. The local host
still verifies the issuer HMAC over the exact action bytes, validates the closed
action contract and scope, owns the replay ledger, executes the runtime action,
and creates the receipt. Thus compromise of a permitted sidecar peer does not
remove the action-level authentication and authorization boundary.

Install this directory as its own wheel and run:

```text
vllm-hust-ride-sidecar --config /absolute/path/sidecar.json
```

The configuration must match `sidecar-config.schema.json`. Certificate, private
key, CA, configuration, and Unix socket paths must be absolute; the private key
must not be accessible by group or other users. Rotate client identities by
replacing the process configuration with a new certificate fingerprint set and
restarting the sidecar. Production certificate issuance, revocation publication,
secret backend integration, audit export, high availability, and external RIDE
deployment remain operator responsibilities and release gates.
