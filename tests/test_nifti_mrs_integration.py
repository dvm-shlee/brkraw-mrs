import json

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Extension

from brkraw_mrs import hook
from nifti_mrs.create_nmrs import nifti_mrs_version


def test_nifti_mrs_header_extension_roundtrip():
    metadata = {
        "ResonantNucleus": "1H",
        "SpectrometerFrequency": 123.4,
        "DwellTime": 0.0005,
        "SpectralWidth": 2000.0,
        "ConversionMethod": "brkraw-mrs",
        "ConversionSoftware": "brkraw-mrs",
        "ConversionSoftwareVersion": "test",
    }
    hdr_ext, dwell = hook._build_hdr_ext(metadata)
    hook._apply_dim_tags(hdr_ext, ("points", "averages", "coils"))

    data = np.zeros((8,), dtype=np.float32)
    affine = np.eye(4, dtype=float)
    img = nib.nifti2.Nifti2Image(data, affine)
    header = img.header

    header.set_qform(affine, code=0)
    header.set_sform(affine, code=0)
    header["pixdim"][4] = float(dwell)

    v_major, v_minor = nifti_mrs_version
    header["intent_name"] = f"mrs_v{v_major}_{v_minor}".encode()
    header.set_xyzt_units(xyz=2, t=8)

    json_s = hdr_ext.to_json()
    json.loads(json_s)
    header.extensions.append(Nifti1Extension(44, json_s.encode("utf-8")))

    assert np.isclose(float(header["pixdim"][4]), dwell)
    intent = header["intent_name"].tobytes().rstrip(b"\x00")
    assert intent == f"mrs_v{v_major}_{v_minor}".encode()
    assert header.extensions[0].get_code() == 44
