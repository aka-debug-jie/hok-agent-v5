# Security policy

## Supported version

Only the current `main` branch is supported.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for security issues. Do not open a public issue that
contains device identifiers, signing identities, private layouts, credentials, unpublished data,
or details that could bypass the fail-closed mobile-testbed boundary.

The public repository intentionally ships without an authorized mobile build identity or calibrated
layout. Missing local identity evidence is expected to disable every device-input path.
