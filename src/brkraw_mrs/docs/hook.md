# brkraw-mrs Hook

BrkRaw converter hook for Bruker single-voxel MRS datasets (PRESS/STEAM/SLASER).
This hook ports the spec2nii Bruker MRS workflow into the BrkRaw hook system.

## Install

```bash
pip install brkraw-mrs
brkraw hook install brkraw-mrs
```

## Use

Convert a ParaVision dataset and write NIfTI-MRS + JSON sidecar:

```bash
brkraw convert \
  /path/to/bruker/PV_dataset \
  --output /path/to/output \
  --sidecar
```

The hook automatically detects single-voxel MRS scans based on the method name
(PRESS/STEAM/SLASER) and uses BrkRaw metadata specs to populate NIfTI-MRS header
extensions.

## Notes

- Specs/rules/transforms are provided under the `brkraw-mrs` namespace.
- Metadata is sourced from `specs/metadata_spec.yaml` and `specs/info_spec.yaml`.
- Outputs include NIfTI-MRS with header extensions plus a JSON sidecar when `--sidecar` is used.
- The conversion metadata and dimension logic follow spec2nii behavior.
