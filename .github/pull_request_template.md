## Problem and resulting behavior

Describe the concrete problem, what changes for a user or researcher, and any related issue.

Change type (select applicable):

- [ ] Bug fix
- [ ] Feature or research implementation
- [ ] Documentation, tests or maintenance

## Validation

List the relevant commands and results, with platform/backend when applicable. For a documentation-only change, describe the checks used. State anything untested and why; a new expensive training run is not required for every change.

For measured research or performance changes, include the config, training budget, evaluation set/sample count, compilation mode, synchronized timing conditions and GPU contention, or link the experiment record. Keep codec reconstruction and generated-output quality separate, and qualify unmatched comparisons.

## Review checks

- [ ] The change preserves applicable forward-equivalence, one-forward-per-level and synchronized-benchmark invariants.
- [ ] Tests and relevant documentation reflect the resulting behavior, or the validation section explains why no updates are needed.
- [ ] New code, data or artifacts have clear provenance and appropriate license/redistribution terms, if applicable.
- [ ] The diff and shared logs contain no credentials, private paths, host identities, cloud account details or costs.

Known limitations or follow-up work:
