# Release Candidate Audit

**Scope:** private pre-release repository for the ActiveMap paper.
**Date:** 2026-08-29.
**Purpose:** verify that the repository is safe to keep private during review
and ready for a later, separately approved public-release pass.

## Verified Locally

| Check | Result |
| --- | --- |
| Tracked large binary/data/checkpoint paths | No tracked raw-data, output, checkpoint, geospatial-binary, or release-artifact paths found. |
| Private infrastructure and credential scan | No match in tracked source/documentation files. Local Git metadata is excluded from the repository. |
| Ignore policy | Ignores environments, credentials, raw/interim/processed/manifests data, run outputs, logs, checkpoints, geospatial binaries, and private operations folders. |
| Documentation entry points | README, reproducibility guide, model/data availability note, dataset protocol, metrics guide, downloaders, smoke script, and bootstrap script all exist. |
| Shell scripts | All release shell scripts pass `bash -n`. |
| Python package source | 140 Python source files pass AST parsing. |
| Packaging metadata | `pyproject.toml` parses as TOML. |
| Diff whitespace | `git diff --check` passes. |

## Remediations In This Audit

1. Restored normal TLS certificate verification in the Inria downloader. The
   downloader now fails rather than bypassing certificate validation when the
   provider endpoint cannot be verified.
2. Reconciled package metadata with the repository license: the project is a
   private pre-release with no public license until the third-party data/model
   license audit is complete.

## Intentional Release Boundary

The repository includes code, schemas, synthetic smoke data, tests, and
construction/evaluation protocols. It deliberately excludes raw third-party
data, pretrained models, paper checkpoints, experiment logs, sealed-test
artifacts, private infrastructure records, and author-identifying operations.
External data and models must be obtained from their original providers under
their own terms.

## Still Required Before Public Release

- Run the full CI workflow in a clean Conda environment. The local base Python
  has CPU PyTorch but lacks the required geospatial stack, and no existing
  isolated ActiveMap environment is available; this audit deliberately did not
  install or alter a user environment.
- Perform a fresh third-party data/model license and redistribution audit.
- Select and add an explicit public software license.
- Add author, citation, and checkpoint-availability metadata after
  anonymization is no longer required.
- Re-run the tracked-file, credential, and large-artifact scan immediately
  before making the repository public.

This audit does not alter the frozen anonymous ICLR submission package or
claim that public reproduction of the sealed paper benchmark is already
available.

## 2026-08-29 Synchronization Update

- Restored `RELEASE_NOTICE.md` from the verified anonymous reviewer-artifact
  snapshot. Its SHA-256 matches the anonymous artifact's notice exactly.
- Aligned the README verification command with the documented CI path:
  complete `pytest` followed by the synthetic smoke workflow.
- Re-ran static checks on the current release candidate: `git diff --check`,
  shell syntax validation, Python AST parsing (154 source/test/tool files),
  and TOML parsing all pass.
- The full test and smoke workflow remain pending on a clean Conda/CI
  environment. This audit intentionally does not install dependencies or
  alter a user environment.
