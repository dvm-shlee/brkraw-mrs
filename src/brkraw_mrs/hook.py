from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Extension

from nifti_mrs.hdr_ext import Hdr_Ext
from nifti_mrs.create_nmrs import nifti_mrs_version

from brkraw.resolver import datatype as datatype_resolver

from . import __version__

logger = logging.getLogger("brkraw-mrs")


def _strip_bruker_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text.strip()


def _first_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    if isinstance(value, np.ndarray) and value.size:
        return value.flat[0]
    return value


def _select_raw_file(scan: Any):
    candidates = [
        ("fid", lambda: scan.file_fid),
        ("rawdata.job0", lambda: scan.file_rawdata_job0),
        ("fid", lambda: scan["fid"]),
        ("rawdata.job0", lambda: scan["rawdata.job0"]),
    ]
    for name, getter in candidates:
        try:
            obj = getter()
        except Exception:
            obj = None
        if obj is not None:
            return name, obj
    return None, None


def _read_bytes(fileobj: Any) -> bytes:
    if fileobj is None:
        return b""
    try:
        if hasattr(fileobj, "seek"):
            fileobj.seek(0)
    except Exception:
        pass
    try:
        return fileobj.read()
    except Exception:
        return bytes(fileobj)


def _resolve_dtype(scan: Any) -> np.dtype:
    dtype_info = datatype_resolver.resolve(scan)
    if dtype_info and "dtype" in dtype_info:
        return np.dtype(dtype_info["dtype"])

    acqp = getattr(scan, "acqp", None)
    byte_order = _strip_bruker_string(acqp.get("BYTORDA") if acqp else None)
    go_format = acqp.get("GO_raw_data_format") if acqp else None
    word_size = acqp.get("ACQ_word_size") if acqp else None

    order = "<" if (byte_order or "").lower().startswith("little") else ">"
    go_mapping = {
        "GO_32BIT_SGN_INT": "i4",
        "GO_16BIT_SGN_INT": "i2",
        "GO_32BIT_FLOAT": "f4",
    }
    if go_format in go_mapping:
        return np.dtype(order + go_mapping[go_format])

    word_size_text = str(word_size or "")
    if "32" in word_size_text:
        return np.dtype(order + "i4")
    if "16" in word_size_text:
        return np.dtype(order + "i2")

    return np.dtype(order + "i4")


def _points_prior_to_echo(method: Any, acqp: Any) -> int:
    dig_shift = _first_value(getattr(method, "PVM_DigShift", None) if method else None)
    if dig_shift is not None:
        try:
            return int(dig_shift)
        except Exception:
            pass
    sw_version = _strip_bruker_string(getattr(acqp, "ACQ_sw_version", None) if acqp else None) or ""
    if "360.1.1" not in sw_version:
        return 0
    rx_filter = getattr(acqp, "ACQ_RxFilterInfo", None) if acqp else None
    if rx_filter is None:
        return 0
    try:
        candidate = rx_filter[0][0]
    except Exception:
        try:
            candidate = rx_filter[0]
        except Exception:
            return 0
    try:
        return int(round(float(candidate)))
    except Exception:
        return 0


def _infer_points(method: Any, acqp: Any) -> Tuple[int, list]:
    candidates = []
    spec_matrix = _first_value(getattr(method, "PVM_SpecMatrix", None) if method else None)
    if spec_matrix:
        try:
            candidates.append(int(spec_matrix))
        except Exception:
            pass
    acq_size = _first_value(getattr(acqp, "ACQ_size", None) if acqp else None)
    if acq_size:
        try:
            acq_size = int(acq_size)
            candidates.append(acq_size * 2)
            candidates.append(acq_size)
        except Exception:
            pass
    if not candidates:
        candidates.append(0)
    return candidates[0], candidates


def _infer_dims(total_complex: int, points_candidates: list, averages: Optional[int], dynamics: Optional[int], coils: Optional[int]) -> Dict[str, int]:
    def _as_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    averages = _as_int(averages)
    dynamics = _as_int(dynamics)
    coils = _as_int(coils)

    for points in points_candidates:
        if points is None or points <= 0:
            continue
        if total_complex % points != 0:
            continue
        rest = total_complex // points
        dims = {"points": points, "averages": averages, "dynamics": dynamics, "coils": coils}
        known_product = 1
        missing = []
        for key in ["averages", "dynamics", "coils"]:
            val = dims[key]
            if val is None:
                missing.append(key)
            else:
                known_product *= max(1, val)
        if rest % known_product != 0:
            continue
        remaining = rest // known_product
        if not missing:
            if known_product == rest:
                return {"points": points, "averages": averages or 1, "dynamics": dynamics or 1, "coils": coils or 1}
            continue
        if len(missing) == 1:
            dims[missing[0]] = remaining
        else:
            # Heuristic: put remaining into averages, keep others as 1
            if "averages" in missing:
                dims["averages"] = remaining
                if "dynamics" in missing:
                    dims["dynamics"] = 1
                if "coils" in missing:
                    dims["coils"] = 1
            elif "dynamics" in missing:
                dims["dynamics"] = remaining
                if "coils" in missing:
                    dims["coils"] = 1
            elif "coils" in missing:
                dims["coils"] = remaining
        return {
            "points": int(points),
            "averages": int(dims["averages"] or 1),
            "dynamics": int(dims["dynamics"] or 1),
            "coils": int(dims["coils"] or 1),
        }

    fallback_points = points_candidates[0] if points_candidates else 0
    return {
        "points": int(fallback_points),
        "averages": int(averages or 1),
        "dynamics": int(dynamics or 1),
        "coils": int(coils or 1),
    }


def _reshape_with_hypotheses(data: np.ndarray, dims: Dict[str, int]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    points = dims["points"]
    averages = dims["averages"]
    dynamics = dims["dynamics"]
    coils = dims["coils"]

    orders = [
        ("points", "averages", "dynamics", "coils"),
        ("points", "dynamics", "averages", "coils"),
        ("points", "averages", "coils", "dynamics"),
        ("points", "coils", "averages", "dynamics"),
    ]
    dim_map = {
        "points": points,
        "averages": averages,
        "dynamics": dynamics,
        "coils": coils,
    }

    for order in orders:
        shape = tuple(dim_map[name] for name in order)
        if np.prod(shape) != data.size:
            continue
        try:
            return data.reshape(shape), order
        except Exception:
            continue
    raise ValueError("Unable to reshape data to expected dimensions.")


def _metadata_from_scan(scan: Any) -> Dict[str, Any]:
    metadata = {}
    try:
        metadata = scan.get_metadata() or {}
    except Exception as exc:
        logger.warning("Failed to resolve metadata spec: %s", exc)
        metadata = {}

    source_dataset = getattr(getattr(scan, "fs", None), "root", None)
    if source_dataset:
        metadata["SourceDataset"] = str(source_dataset)

    scan_id = getattr(scan, "scan_id", None)
    if scan_id is not None:
        metadata["ScanIdentifier"] = str(scan_id)

    metadata["ConversionSoftware"] = "brkraw-mrs"
    metadata["ConversionSoftwareVersion"] = __version__

    return metadata


def _build_hdr_ext(metadata: Dict[str, Any]) -> Tuple[Hdr_Ext, float]:
    nucleus = metadata.get("ResonantNucleus")
    if not nucleus:
        nucleus = ["1H"]
    if isinstance(nucleus, str):
        nucleus = [nucleus]

    freq = metadata.get("SpectrometerFrequency")
    if isinstance(freq, (int, float)):
        freq = [float(freq)]
    if not freq:
        freq = [0.0]

    hdr_ext = Hdr_Ext(freq, nucleus)

    echo = metadata.get("EchoTime")
    if echo is not None:
        hdr_ext.set_standard_def("EchoTime", float(echo))
    tr = metadata.get("RepetitionTime")
    if tr is not None:
        hdr_ext.set_standard_def("RepetitionTime", float(tr))
    sw = metadata.get("SpectralWidth")
    if sw is not None:
        hdr_ext.set_standard_def("SpectralWidth", float(sw))

    hdr_ext.set_standard_def("ConversionMethod", f"brkraw-mrs v{__version__}")
    hdr_ext.set_standard_def(
        "ConversionTime",
        datetime.now().isoformat(sep="T", timespec="milliseconds"),
    )
    hdr_ext.set_standard_def("Manufacturer", "Bruker")
    if metadata.get("SoftwareVersions") is not None:
        hdr_ext.set_standard_def("SoftwareVersions", metadata["SoftwareVersions"])
    if metadata.get("SequenceName") is not None:
        hdr_ext.set_standard_def("SequenceName", metadata["SequenceName"])
    if metadata.get("PatientName") is not None:
        hdr_ext.set_standard_def("PatientName", metadata["PatientName"])
    if metadata.get("TxOffset") is not None:
        hdr_ext.set_standard_def("TxOffset", float(metadata["TxOffset"]))
    if metadata.get("OriginalFile") is not None:
        hdr_ext.set_standard_def("OriginalFile", metadata["OriginalFile"])
    hdr_ext.set_standard_def("kSpace", [False, False, False])

    hdr_ext.set_user_def("ConversionSoftware", "brkraw-mrs", "Conversion software name")
    hdr_ext.set_user_def(
        "ConversionSoftwareVersion",
        __version__,
        "Conversion software version",
    )
    if "SourceDataset" in metadata:
        hdr_ext.set_user_def(
            "SourceDataset",
            metadata["SourceDataset"],
            "Original dataset path",
        )
    if "ScanIdentifier" in metadata:
        hdr_ext.set_user_def(
            "ScanIdentifier",
            metadata["ScanIdentifier"],
            "BrkRaw scan identifier",
        )

    dwell = metadata.get("DwellTime")
    if dwell is None and sw:
        try:
            dwell = 1.0 / float(sw)
        except Exception:
            dwell = None
    if dwell is None:
        dwell = 0.0
    return hdr_ext, float(dwell)


def _apply_dim_tags(hdr_ext: Hdr_Ext, dim_names: Tuple[str, ...]) -> None:
    tag_map = {
        "averages": "DIM_DYN",
        "dynamics": "DIM_DYN",
        "coils": "DIM_COIL",
    }
    for idx, name in enumerate(dim_names[1:], start=0):
        tag = tag_map.get(name, f"DIM_USER_{idx}")
        hdr_ext.set_dim_info(idx, tag)


def get_dataobj(self, reco_id: Optional[int] = None):
    _ = reco_id
    return _load_press_data(self)[0]


def get_affine(self, reco_id: Optional[int] = None, *, decimals: Optional[int] = None):
    _ = reco_id
    _ = decimals
    return np.eye(4, dtype=float)


def convert(self, reco_id: Optional[int] = None, **kwargs):
    _ = reco_id
    _ = kwargs
    data, dim_order, metadata = _load_press_data(self)

    hdr_ext, dwell = _build_hdr_ext(metadata)
    _apply_dim_tags(hdr_ext, dim_order)

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
    header.extensions.append(Nifti1Extension(44, json_s.encode("utf-8")))

    return img


def _load_press_data(scan: Any) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, Any]]:
    method = getattr(scan, "method", None)
    acqp = getattr(scan, "acqp", None)

    method_name = getattr(method, "Method", None) if method else None
    method_text = _strip_bruker_string(method_name) or ""
    if "PRESS" not in method_text.upper():
        logger.warning("Skipping scan %s: method does not look like PRESS (%s)", getattr(scan, "scan_id", "?"), method_text)
        raise ValueError("Not a PRESS SVS scan.")

    file_name, file_obj = _select_raw_file(scan)
    if file_obj is None:
        logger.warning("Skipping scan %s: no fid/rawdata file found.", getattr(scan, "scan_id", "?"))
        raise FileNotFoundError("No fid/rawdata file found.")

    raw = _read_bytes(file_obj)
    logger.info("Using raw file: %s", file_name)
    logger.info("Raw file size (bytes): %s", len(raw))

    dtype = _resolve_dtype(scan)
    logger.info("Resolved dtype: %s", dtype)

    data = np.frombuffer(raw, dtype=dtype)
    if data.size % 2 != 0:
        raise ValueError("FID data length is not even; cannot form complex pairs.")

    real = data[0::2]
    imag = data[1::2]
    complex_data = real.astype(np.float32, copy=False) + 1j * imag.astype(np.float32, copy=False)

    points_default, points_candidates = _infer_points(method, acqp)
    averages = _first_value(getattr(method, "PVM_NAverages", None) if method else None)
    dynamics = _first_value(getattr(method, "PVM_NRepetitions", None) if method else None)
    coils = _first_value(getattr(method, "PVM_EncNReceivers", None) if method else None)

    dims = _infer_dims(
        total_complex=complex_data.size,
        points_candidates=points_candidates,
        averages=averages,
        dynamics=dynamics,
        coils=coils,
    )

    expected = dims["points"] * dims["averages"] * dims["dynamics"] * dims["coils"]
    logger.info(
        "Derived dims points=%s averages=%s dynamics=%s coils=%s (expected samples=%s)",
        dims["points"],
        dims["averages"],
        dims["dynamics"],
        dims["coils"],
        expected,
    )
    logger.info("Actual complex sample count=%s", complex_data.size)
    expected_bytes = complex_data.size * 2 * dtype.itemsize
    logger.info("Expected raw byte size=%s (dtype=%s)", expected_bytes, dtype)

    reshaped, order = _reshape_with_hypotheses(complex_data, dims)
    logger.info("Reshape order: %s", order)

    points_shift = _points_prior_to_echo(method, acqp)
    if points_shift:
        if points_shift >= reshaped.shape[0]:
            raise ValueError("points_prior_to_echo exceeds available points.")
        reshaped = reshaped[points_shift:, ...]
        logger.info("Removed points prior to echo: %s", points_shift)

    reshaped = reshaped.conj()

    data = np.expand_dims(reshaped, axis=0)
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)

    metadata = _metadata_from_scan(scan)
    software = getattr(acqp, "ACQ_sw_version", None) if acqp else None
    if software is not None:
        metadata["SoftwareVersions"] = str(software)
    seq_name = (getattr(method, "Method", None) if method else None) or (getattr(method, "PULPROG", None) if method else None)
    if seq_name is not None:
        metadata["SequenceName"] = _strip_bruker_string(seq_name)
    tx_offset = _first_value(getattr(method, "PVM_FrqWorkOffsetPpm", None) if method else None)
    if tx_offset is not None:
        try:
            metadata["TxOffset"] = float(tx_offset)
        except Exception:
            pass
    source_dataset = metadata.get("SourceDataset")
    if source_dataset is not None:
        metadata["OriginalFile"] = [str(source_dataset)]
    if "SpectrometerFrequency" not in metadata:
        freq = getattr(acqp, "BF1", None) if acqp else None
        if freq is not None:
            metadata["SpectrometerFrequency"] = [float(_first_value(freq))]
    if "ResonantNucleus" not in metadata:
        nucleus = _strip_bruker_string(getattr(acqp, "NUCLEUS", None) if acqp else None)
        if nucleus:
            metadata["ResonantNucleus"] = [nucleus]

    if "SpectralWidth" not in metadata:
        sw = getattr(method, "PVM_SpecSWH", None) if method else None
        if sw is not None:
            metadata["SpectralWidth"] = float(_first_value(sw))

    if "DwellTime" not in metadata:
        sw = metadata.get("SpectralWidth")
        if sw:
            metadata["DwellTime"] = 1.0 / float(sw)

    for key in ["ResonantNucleus", "SpectrometerFrequency", "SpectralWidth", "DwellTime", "EchoTime", "RepetitionTime"]:
        if key not in metadata or metadata[key] in (None, ""):
            logger.warning("Missing metadata key: %s", key)

    return data.astype(np.complex64, copy=False), order, metadata


HOOK = {
    "get_dataobj": get_dataobj,
    "get_affine": get_affine,
    "convert": convert,
}
