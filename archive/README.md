# Archive

The archive is not a trash folder. Files placed here are retained for audit, provenance, or historical
comparison, but they are not canonical entrypoints for the current research results.

Before using an archived file:

1. Check the migration map in `docs/repository_audit/legacy_migration_map.csv`.
2. Check whether a canonical replacement exists.
3. Confirm that the archived file is appropriate for historical comparison rather than final analysis.

Current cleanup policy:

- Do not move unknown files into archive.
- Do not archive frozen final artifacts.
- Do not archive outputs that are still used by V1/V2 comparison, manifests, or downstream audit.
- Preserve SHA-256 before and after any `git mv`.
