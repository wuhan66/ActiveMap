# Model And Data Availability

## Included In This Release Candidate

- ActiveMap source code, command-line interface, schemas, smoke configs, and
  curated tests.
- Documentation for dataset construction and final-map metrics.
- Synthetic-data smoke generation with no download, credential, or checkpoint
  requirement.

## Deliberately Excluded

- Raw SpaceNet 7, MUNO21, SpaceNet 8, ArgoTweak, Inria, and third-party data.
- Third-party perception repositories and pretrained weights.
- Paper checkpoints, raw execution logs, server paths, credentials, and sealed
  test manifests.
- Private infrastructure configuration and author-identifying operational
  records.

## Access Rules

External data and model assets must be acquired from their original providers
under the providers' terms. The repository does not redistribute them. The
release candidate verifies the executable pipeline through synthetic
smoke data; it does not purport to reproduce the sealed paper benchmark without
the original restricted inputs.

Before a public release, publish a checkpoint availability matrix, immutable
result manifests where redistribution is permitted, author and citation
metadata, and a final data-license audit.
