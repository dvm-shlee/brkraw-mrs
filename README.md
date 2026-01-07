# brkraw-mrs

BrkRaw converter hook package for Bruker PRESS SVS (MRS) datasets. This project
ports the Bruker MRS workflow from spec2nii and packages it as a BrkRaw hook so
it can be installed and used through BrkRaw's hook tooling.

## What It Does

- Detects PRESS SVS scans (method contains PRESS) and reads raw spectroscopy data
  from `fid` or `rawdata.job0`.
- Converts interleaved real/imag to complex, infers dimensions, and writes
  NIfTI-MRS with a header extension JSON (NIfTI-2 preferred).
- Writes a JSON sidecar with NIfTI-MRS metadata for convenience.

Data ordering for SVS is:

```
(1, 1, 1, n_points, n_averages?, n_dynamics?, n_coils?)
```

If a dimension is missing or equals 1, it is omitted.

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

Legacy install script (kept for older workflows):

```bash
brkraw-mrs-install
```

## BrkRaw Usage

```bash
brkraw convert \
  --path /path/to/bruker/PV_dataset \
  --output /path/to/output \
  --sidecar
```

## Attribution and License

This hook is a port of the spec2nii Bruker MRS workflow. See `NOTICE` for
attribution details and `LICENSE` for terms.
