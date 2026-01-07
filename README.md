# brkraw-mrs

BrkRaw converter hook package that adds single-voxel MRS conversion support to
the brkraw ecosystem. This project ports the Bruker MRS workflow from spec2nii
and packages it as a BrkRaw hook so it can be installed and used through
BrkRaw's hook tooling.

## What It Does

- Detects single-voxel MRS scans (PRESS/STEAM/SLASER by method name) and reads
  raw spectroscopy data from `fid` or `rawdata.job0`.
- Converts interleaved real/imag to complex, infers dimensions, and writes
  NIfTI-MRS with a header extension JSON (NIfTI-2 preferred).
- Writes a JSON sidecar with NIfTI-MRS metadata for convenience.

The goal is to make single-voxel MRS data usable with the same downstream tools
as MRI, so MRS workflows can benefit from the broader BrkRaw ecosystem.

## Install (BrkRaw Hook Standard)

Install the Python package, then install the hook assets into BrkRaw:

```bash
pip install brkraw-mrs
brkraw hook install brkraw-mrs
```

To install by entrypoint name:

```bash
brkraw hook install mrs
```

To view the packaged hook docs:

```bash
brkraw hook docs brkraw-mrs --render
```

## BrkRaw Usage

```bash
brkraw convert \
  /path/to/bruker/PV_dataset \
  --output /path/to/output \
  --sidecar
```

## Support Matrix

Tested datasets are based on the Bruker ParaVision standard datasets:
https://www.bruker.com/protected/en/services/communities/pci-community/paravision-versions/paravision-standard-datasets.html

- PV360 3.5 PRESS: conversion tested
- PV360 3.6 PRESS: conversion tested
- PV360 3.7 PRESS: conversion tested
- Older ParaVision versions: not tested
- Other MRS sequences (STEAM, SLASER, others): not tested

Notes:
Data ordering for PRESS SVS is assumed as:

```
(1, 1, 1, n_points, n_averages?, n_dynamics?, n_coils?)
```

If a dimension is missing or equals 1, it is omitted.

## Contributing

This repository is shared as a prototype. The core BrkRaw maintainers are
primarily focused on MRI conversion and do not have deep MRS domain expertise.
We welcome MRS researchers and developers to validate the workflow and guide
ongoing development so it can evolve beyond a single-voxel prototype. Please
see `CONTRIBUTING.md` and start a discussion at:
https://github.com/orgs/BrkRaw/discussions/categories/brkraw-mrs

## Attribution and License

This hook is a port of the spec2nii Bruker MRS workflow. See `NOTICE` for
attribution details and `LICENSE` for terms.
