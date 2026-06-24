# Security Policy

## Supported Versions

TileMEM / TilePO is currently a research artifact and public reproducibility
release. Security fixes are provided for the active `main` branch and the latest
v0.1.x public artifact.

| Version / Branch | Supported |
| --- | --- |
| main | Yes |
| v0.1.x / v0.1.1 | Yes |
| older snapshots | No |

Performance regressions, benchmark disagreements, or workload-specific negative
results are not treated as security vulnerabilities unless they expose data,
execute unintended code, bypass isolation, or corrupt user files.

## Reporting a Vulnerability

Please do not disclose security vulnerabilities in public GitHub issues.

To report a vulnerability, use one of the following private channels:

1. GitHub private vulnerability reporting, if enabled for this repository.
2. Otherwise, open a minimal GitHub issue titled `Security report request`
   without exploit details, and the maintainer will arrange a private channel.

Please include:

- affected commit, tag, or release;
- operating system and Python/CUDA environment;
- steps to reproduce;
- impact assessment;
- whether the issue allows code execution, path traversal, data disclosure,
  artifact tampering, denial of service, or unsafe model/checkpoint handling.

Expected response:

- acknowledgement within 7 days;
- initial triage within 14 days;
- accepted vulnerabilities will receive a fix, mitigation, or documented
  workaround as appropriate;
- declined reports will receive a short explanation.

For high-severity issues, please allow reasonable time for a fix before public
disclosure.
