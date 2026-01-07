# brkraw-mrs Hook

BrkRaw converter hook for Bruker PRESS SVS (MRS) datasets. This hook ports the
spec2nii Bruker MRS workflow into the BrkRaw hook system.

## Install

```bash
pip install brkraw-mrs
brkraw hook install brkraw-mrs
```

## Use

```bash
brkraw convert \
  /path/to/bruker/PV_dataset \
  --output /path/to/output \
  --sidecar
```

## Notes

- The hook installs specs, rules, and transforms under a `brkraw-mrs` namespace.
- The conversion metadata and dimension logic follow spec2nii behavior.
