# brkraw-mrs

A BrkRaw converter hook that adds single-voxel MRS support to the BrkRaw ecosystem.
This package ports the Bruker MRS workflow from spec2nii and exposes it as a
first-party BrkRaw hook.

## What it does

- Detects single-voxel MRS scans (PRESS, STEAM, SLASER) from ParaVision metadata
- Reads spectroscopy data from `fid` or `rawdata.job0`
- Converts data to NIfTI-MRS (NIfTI-2 preferred) with JSON header extensions

The goal is to make single-voxel MRS data usable within standard MRI analysis
workflows supported by BrkRaw.

## Install (BrkRaw Hook Standard)

Install the package and register the hook with BrkRaw:

```bash
pip install brkraw-mrs
brkraw hook install brkraw-mrs
```

To install by entrypoint name:

```bash
brkraw hook install mrs
```

To view the hook documentation:

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

## Support

Tested with Bruker ParaVision standard datasets:

- PV360 3.5-3.7 (PRESS)
- Other MRS sequences and older ParaVision versions are untested

Notes:
Data ordering for PRESS SVS is assumed as:

```plane
(1, 1, 1, n_points, n_averages?, n_dynamics?, n_coils?)
```

If a dimension is missing or equals 1, it is omitted.

## Contributing

This repository provides a foundational implementation of single-voxel MRS
support within the BrkRaw framework. Contributions from MRS researchers are
encouraged to help validate, refine, and extend the workflow across sequences,
acquisition schemes, and ParaVision versions.

## Attribution and License

This hook is a port of the spec2nii Bruker MRS workflow. See `NOTICE` for
attribution details and `LICENSE` for terms.
