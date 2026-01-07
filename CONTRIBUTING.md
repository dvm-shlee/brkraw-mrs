# Contributing to brkraw-mrs

brkraw-mrs is an extension that adds single-voxel MRS conversion to BrkRaw so
MRS data can be handled with the same tools and workflows used for MRI. The goal
is to provide a unified converter that makes MRS data more accessible to the
broader imaging community.

The core BrkRaw maintainers focus on MRI conversion and do not have deep MRS
domain expertise. This repository is intentionally published as a prototype,
and we are looking for MRS researchers and developers to help validate the
pipeline, expand sequence coverage, and shape the roadmap.

If you are interested in contributing, please start a discussion here:
https://github.com/orgs/BrkRaw/discussions/categories/brkraw-mrs

## How to Contribute

1. Start a discussion or open an issue describing your dataset, sequence, and
   expected behavior.
2. Fork the repo and create a feature branch.
3. Keep changes focused and scoped to a single goal.
4. Submit a PR with a clear description of the changes and how they were tested.

## What We Need Most

- Validation on additional datasets and ParaVision versions
- Sequence coverage and metadata improvements
- Test cases and reproducible evaluation scripts
- Documentation and usage examples for MRS-specific workflows

## Data and Reproducibility

- Provide anonymized or public datasets when possible.
- Include acquisition details (ParaVision version, sequence name, key params).
- Prefer minimal, reproducible scripts that show inputs and expected outputs.

## Development Notes

- Metadata rules live under `brkraw-mrs/src/brkraw_mrs/specs`.
- Hook rules live under `brkraw-mrs/src/brkraw_mrs/rules`.
- Transform helpers live under `brkraw-mrs/src/brkraw_mrs/transforms`.
- Conversion logic lives in `brkraw-mrs/src/brkraw_mrs/hook.py`.

## Review Expectations

- PRs should include a short test plan and any validation data used.
- Behavior changes should update `brkraw-mrs/README.md` when relevant.
- If you add a new sequence, update the support matrix and rules.
