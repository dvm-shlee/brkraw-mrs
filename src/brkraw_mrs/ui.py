from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk

import nibabel as nib

from .hook import _load_mrs_data

logger = logging.getLogger("brkraw.mrs")

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class MRSPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, *, app: Any) -> None:
        super().__init__(parent)
        self._app = app
        self._mrs_data: Optional[np.ndarray] = None
        self._mrs_order: Optional[Tuple[str, ...]] = None
        self._mrs_meta: Dict[str, Any] = {}

        self._status_var = tk.StringVar(value="")
        self._underlay_var = tk.StringVar(value="Underlay: none")

        self._lb_var = tk.DoubleVar(value=0.0)
        self._phase_var = tk.DoubleVar(value=0.0)
        self._shift_var = tk.DoubleVar(value=0.0)

        self._avg_dim_vars: Dict[str, tk.BooleanVar] = {}
        self._avg_dim_frame: Optional[ttk.Frame] = None

        self._spec_canvas: Optional[Any] = None
        self._spec_figure: Optional[Any] = None
        self._spec_axes: Optional[Any] = None
        self._spec_toolbar: Optional[Any] = None
        self._last_message: Optional[str] = None
        self._plot_refresh_after: Optional[str] = None

        self._underlay_canvas: Optional[tk.Canvas] = None
        self._underlay_image: Optional[Any] = None
        self._underlay_plane_var = tk.StringVar(value="xy")
        self._underlay_slice_var = tk.IntVar(value=0)
        self._underlay_slice_label = tk.StringVar(value="Slice: -")
        self._underlay_slice_scale: Optional[tk.Scale] = None
        self._underlay_data: Optional[np.ndarray] = None
        self._underlay_affine: Optional[np.ndarray] = None
        self._underlay_affine_scanner: Optional[np.ndarray] = None
        self._underlay_res: Optional[np.ndarray] = None
        self._underlay_res_scanner: Optional[np.ndarray] = None
        self._underlay_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._underlay_refresh_after: Optional[str] = None
        self._underlay_scan_id: Optional[int] = None
        self._underlay_reco_id: Optional[int] = None
        self._last_scan_id: Optional[int] = None
        self._underlay_space_label = "scanner"

        self._tooltips: list[_ToolTip] = []

        self._make_ui()

    def _make_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(top, text="MRS Spectrum").grid(row=0, column=0, sticky="w")

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=320)
        body.rowconfigure(0, weight=1)

        spectrum = ttk.Frame(body)
        spectrum.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        spectrum.columnconfigure(0, weight=1)
        spectrum.rowconfigure(0, weight=1)

        controls_row = 1
        if HAS_MPL:
            self._spec_figure = Figure(figsize=(4, 3), dpi=100)
            self._spec_figure.set_constrained_layout(False)
            self._spec_axes = self._spec_figure.add_subplot(111)
            self._spec_axes.set_facecolor("#101010")
            self._spec_figure.patch.set_facecolor("#101010")
            self._spec_canvas = FigureCanvasTkAgg(self._spec_figure, master=spectrum)
            self._spec_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self._spec_canvas.get_tk_widget().configure(background="#101010")
            self._spec_canvas.get_tk_widget().bind("<Configure>", self._on_canvas_resize)
            toolbar_frame = ttk.Frame(spectrum)
            toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self._spec_toolbar = NavigationToolbar2Tk(self._spec_canvas, toolbar_frame)
            self._spec_toolbar.update()
            controls_row = 2
        else:
            self._spec_canvas = tk.Canvas(spectrum, background="#101010", highlightthickness=0)
            self._spec_canvas.grid(row=0, column=0, sticky="nsew")
            self._spec_canvas.bind("<Configure>", self._on_canvas_resize)

        controls = ttk.Frame(spectrum)
        controls.grid(row=controls_row, column=0, sticky="ew", pady=(6, 0))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Spectrum preprocessing").grid(row=0, column=0, columnspan=2, sticky="w")

        self._avg_dim_frame = ttk.Frame(controls)
        self._avg_dim_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        ttk.Label(controls, text="Line broadening (Hz)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        lb_entry = ttk.Entry(controls, textvariable=self._lb_var)
        lb_entry.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self._add_tooltip(lb_entry, "Apply exponential line broadening in time domain (Hz).")

        ttk.Label(controls, text="Phase0 (deg)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        phase_entry = ttk.Entry(controls, textvariable=self._phase_var)
        phase_entry.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        self._add_tooltip(phase_entry, "Zero-order phase correction in degrees.")

        ttk.Label(controls, text="Freq shift (Hz)").grid(row=4, column=0, sticky="w", pady=(6, 0))
        shift_entry = ttk.Entry(controls, textvariable=self._shift_var)
        shift_entry.grid(row=4, column=1, sticky="ew", pady=(6, 0))
        self._add_tooltip(shift_entry, "Frequency shift applied to the spectrum (Hz).")

        ttk.Button(controls, text="Apply", command=self._schedule_plot_refresh).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        underlay = ttk.Frame(body, width=320)
        underlay.grid(row=0, column=1, sticky="nsew")
        underlay.grid_propagate(False)
        underlay.columnconfigure(0, weight=1)
        underlay.rowconfigure(1, weight=1)

        underlay_top = ttk.Frame(underlay)
        underlay_top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(underlay_top, text="Plane").pack(side=tk.LEFT, padx=(0, 6))
        plane_combo = ttk.Combobox(
            underlay_top,
            state="readonly",
            values=["xy", "xz", "yz"],
            width=6,
            textvariable=self._underlay_plane_var,
        )
        plane_combo.pack(side=tk.LEFT)
        plane_combo.bind("<<ComboboxSelected>>", self._on_plane_change)
        self._add_tooltip(plane_combo, "Plane orientation for the underlay view.")
        ttk.Label(underlay_top, textvariable=self._underlay_slice_label).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(underlay_top, text="Select underlay…", command=self._open_underlay_selector).pack(
            side=tk.RIGHT
        )
        ttk.Label(underlay_top, textvariable=self._underlay_var).pack(side=tk.RIGHT, padx=(0, 10))

        self._underlay_canvas = tk.Canvas(underlay, background="#0f0f0f", highlightthickness=0, width=300, height=300)
        self._underlay_canvas.grid(row=1, column=0, sticky="nsew")
        self._underlay_canvas.bind("<Configure>", self._on_underlay_resize)

        slice_scale = tk.Scale(
            underlay,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=True,
            command=lambda _v: self._schedule_underlay_refresh(),
            length=180,
        )
        slice_scale.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        slice_scale.configure(variable=self._underlay_slice_var)
        self._underlay_slice_scale = slice_scale
        self._add_tooltip(slice_scale, "Slice index for the selected plane.")

        status = ttk.Label(self, textvariable=self._status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def get_voxel_info(self) -> tuple[Optional[Any], Optional[Any]]:
        return self._mrs_meta.get("VoxelPosition"), self._mrs_meta.get("VoxelSize")

    def refresh_from_viewer(self) -> None:
        scan = getattr(self._app, "_scan", None)
        if scan is None:
            self._mrs_data = None
            self._mrs_order = None
            self._mrs_meta = {}
            self._clear_underlay()
            self._show_message("No scan selected.")
            return
        try:
            data, order, metadata = _load_mrs_data(scan)
        except Exception as exc:
            self._mrs_data = None
            self._mrs_order = None
            self._mrs_meta = {}
            self._show_message(f"Not an MRS scan:\n{exc}")
            return
        self._mrs_data = data
        self._mrs_order = order
        self._mrs_meta = metadata
        self._refresh_avg_dim_controls()
        self._schedule_plot_refresh()
        self._sync_underlay_with_viewer(scan)

    def _refresh_plot(self) -> None:
        self._plot_refresh_after = None
        if self._mrs_data is None or self._mrs_order is None:
            self._show_message("No MRS data.")
            return
        fid, dwell = self._process_fid()
        if fid is None or dwell is None:
            self._show_message("Failed to process FID.")
            return
        spectra, freq = self._build_spectra(fid, dwell)
        self._plot_spectrum(spectra, freq)
        status = f"Points: {fid.shape[0]}  Dwell: {dwell:.6f}s"
        voxel_pos = self._format_vector(self._mrs_meta.get("VoxelPosition"))
        voxel_size = self._format_vector(self._mrs_meta.get("VoxelSize"))
        if voxel_pos:
            status += f"  Voxel pos: {voxel_pos} mm"
        if voxel_size:
            status += f"  Size: {voxel_size} mm"
        self._status_var.set(status)

    def _process_fid(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        data = self._mrs_data
        if data is None:
            return None, None
        data = np.asarray(data)
        if data.ndim < 4:
            return None, None
        data = data.reshape(data.shape[3:])
        order = self._mrs_order or tuple(f"dim{idx}" for idx in range(data.ndim))
        order = order[: data.ndim]
        for axis in range(1, data.ndim):
            label = order[axis] if axis < len(order) else f"dim{axis}"
            var = self._avg_dim_vars.get(label)
            if var is not None and var.get():
                data = data.mean(axis=axis, keepdims=True)
        fid = data

        dwell = float(self._mrs_meta.get("DwellTime") or 0.0) or None
        if dwell is None:
            dwell = 1.0

        t = np.arange(fid.shape[0]) * dwell
        lb = float(self._lb_var.get() or 0.0)
        if lb:
            fid = fid * np.exp(-lb * np.pi * t)
        phase = np.deg2rad(float(self._phase_var.get() or 0.0))
        if phase:
            fid = fid * np.exp(1j * phase)
        shift = float(self._shift_var.get() or 0.0)
        if shift:
            fid = fid * np.exp(1j * 2 * np.pi * shift * t)

        return fid, dwell

    def _build_spectra(self, fid: np.ndarray, dwell: float) -> Tuple[list[np.ndarray], np.ndarray]:
        data = np.asarray(fid)
        if data.ndim == 1:
            spectra = [np.fft.fftshift(np.fft.fft(data))]
        else:
            spectra = []
            for idx in product(*[range(dim) for dim in data.shape[1:]]):
                series = data[(slice(None),) + idx]
                spectra.append(np.fft.fftshift(np.fft.fft(series)))
        freq = np.fft.fftshift(np.fft.fftfreq(data.shape[0], d=dwell))
        return spectra, freq

    def _plot_spectrum(self, spectra: list[np.ndarray], freq: np.ndarray) -> None:
        if HAS_MPL and self._spec_axes is not None and self._spec_canvas is not None:
            self._spec_axes.clear()
            self._spec_axes.set_facecolor("#101010")
            if self._spec_figure is not None:
                self._spec_figure.patch.set_facecolor("#101010")
            self._spec_axes.tick_params(colors="#cccccc", labelsize=8)
            for spectrum in spectra:
                self._spec_axes.plot(freq, spectrum.real, linewidth=0.8, alpha=0.6, color="#66ccff")
            self._spec_axes.set_xlabel("Frequency (Hz)", color="#cccccc", labelpad=6)
            self._spec_axes.set_ylabel("Amplitude", color="#cccccc", labelpad=6)
            self._spec_axes.grid(False)
            self._spec_axes.set_position([0.08, 0.22, 0.9, 0.72])
            self._spec_figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.22)
            try:
                self._spec_canvas.draw()
            except Exception:
                self._spec_canvas.draw_idle()
            return
        self._render_message("Install matplotlib to view spectrum.")

    def _format_vector(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            items = list(value)
        else:
            items = [value]
        if not items:
            return None
        try:
            return "(" + ", ".join(f"{float(v):.2f}" for v in items) + ")"
        except Exception:
            return "(" + ", ".join(str(v) for v in items) + ")"

    def _show_message(self, message: str) -> None:
        self._status_var.set(message)
        self._last_message = message
        self._render_message(message)

    def _render_message(self, message: str) -> None:
        if HAS_MPL and self._spec_axes is not None and self._spec_canvas is not None:
            self._spec_axes.clear()
            self._spec_axes.set_facecolor("#101010")
            self._spec_axes.text(
                0.5,
                0.5,
                message,
                color="#ff4444",
                ha="center",
                va="center",
                transform=self._spec_axes.transAxes,
            )
            self._spec_axes.set_xticks([])
            self._spec_axes.set_yticks([])
            self._spec_canvas.draw_idle()
            return
        canvas = self._spec_canvas if isinstance(self._spec_canvas, tk.Canvas) else None
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_text(
            width // 2,
            height // 2,
            anchor="center",
            fill="#ff4444",
            text=message,
            font=("TkDefaultFont", 11, "bold"),
        )

    def _open_underlay_selector(self) -> None:
        scan = getattr(self._app, "_scan", None)
        scan_id = self._underlay_scan_id or getattr(scan, "scan_id", None)
        reco_id = self._underlay_reco_id or getattr(self._app, "_current_reco_id", None)
        UnderlaySelector(
            self,
            app=self._app,
            on_select=self._on_underlay_selected,
            selected_scan_id=scan_id,
            selected_reco_id=reco_id,
        )

    def _on_underlay_selected(self, label: str, scan: Any, reco_id: int) -> None:
        self._underlay_var.set(label)
        self._set_underlay(scan, reco_id)

    def _set_underlay(self, scan: Any, reco_id: int) -> None:
        try:
            dataobj = scan.get_dataobj(reco_id=reco_id)
            affine_raw = scan.get_affine(reco_id=reco_id, space="raw")
            affine_scanner = scan.get_affine(reco_id=reco_id, space="scanner")
        except Exception as exc:
            self._show_underlay_message(f"Underlay load failed:\n{exc}")
            return
        if dataobj is None or affine_raw is None:
            self._show_underlay_message("Underlay data unavailable.")
            return
        if isinstance(dataobj, tuple):
            dataobj = dataobj[0]
        data = np.asarray(dataobj)
        while data.ndim > 3:
            data = data[..., 0]
        if data.ndim < 3:
            self._show_underlay_message("Underlay data is not 3D.")
            return
        self._underlay_scan_id = getattr(scan, "scan_id", None)
        try:
            self._underlay_reco_id = int(reco_id)
        except Exception:
            self._underlay_reco_id = None
        if isinstance(affine_scanner, tuple):
            affine_scanner = affine_scanner[0]
        if affine_scanner is not None:
            self._underlay_space_label = "scanner"
            data, affine_display = self._orient_underlay(data, np.asarray(affine_scanner))
        else:
            self._underlay_space_label = "raw"
            data, affine_display = self._orient_underlay(data, np.asarray(affine_raw))
        self._underlay_data = data
        self._underlay_affine = affine_display
        self._underlay_affine_scanner = None
        self._underlay_res = np.linalg.norm(self._underlay_affine[:3, :3], axis=0)
        self._underlay_res_scanner = None
        self._underlay_box = self._resolve_voxel_box()
        self._update_underlay_slice_range()
        self._schedule_underlay_refresh()
        self._underlay_var.set(f"space: {self._underlay_space_label}")

    def _clear_underlay(self) -> None:
        self._underlay_data = None
        self._underlay_affine = None
        self._underlay_affine_scanner = None
        self._underlay_res = None
        self._underlay_res_scanner = None
        self._underlay_box = None
        self._underlay_scan_id = None
        self._underlay_reco_id = None
        self._underlay_var.set("Underlay: none")
        if self._underlay_canvas is not None:
            self._underlay_canvas.delete("all")
        self._underlay_slice_label.set("Slice: -")

    def _sync_underlay_with_viewer(self, scan: Any) -> None:
        scan_id = getattr(scan, "scan_id", None)
        if scan_id is None:
            self._clear_underlay()
            return
        if scan_id != self._last_scan_id:
            self._clear_underlay()
            self._last_scan_id = scan_id
        current_reco = getattr(self._app, "_current_reco_id", None)
        if self._underlay_scan_id == scan_id and self._underlay_reco_id is not None:
            return
        if current_reco is None:
            return
        self._underlay_var.set(f"{scan_id:03d} :: {int(current_reco):03d} :: {self._underlay_space_label}")
        self._set_underlay(scan, current_reco)

    def _update_underlay_slice_range(self) -> None:
        if self._underlay_data is None:
            self._underlay_slice_var.set(0)
            self._underlay_slice_label.set("Slice: -")
            return
        axis = self._plane_axis()
        size = int(self._underlay_data.shape[axis])
        current = int(self._underlay_slice_var.get() or 0)
        if self._underlay_box is not None:
            mins, maxs = self._underlay_box
            center = int(round((mins[axis] + maxs[axis]) / 2))
            current = center
        if current < 0 or current >= size:
            current = size // 2
        if self._underlay_slice_scale is not None:
            self._underlay_slice_scale.configure(from_=0, to=max(size - 1, 0))
        self._underlay_slice_var.set(current)
        self._underlay_slice_label.set(f"Slice: {current + 1}/{size}")

    def _plane_axis(self) -> int:
        plane = self._underlay_plane_var.get()
        if plane == "xz":
            return 1
        if plane == "yz":
            return 0
        return 2

    def _on_plane_change(self, *_: object) -> None:
        self._update_underlay_slice_range()
        self._schedule_underlay_refresh()

    def _on_underlay_resize(self, _event: tk.Event) -> None:
        self._schedule_underlay_refresh()

    def _schedule_underlay_refresh(self) -> None:
        if self._underlay_refresh_after is not None:
            try:
                self.after_cancel(self._underlay_refresh_after)
            except Exception:
                pass
            self._underlay_refresh_after = None
        self._underlay_refresh_after = self.after_idle(self._render_underlay)

    def _show_underlay_message(self, message: str) -> None:
        if self._underlay_canvas is None:
            return
        self._underlay_canvas.delete("all")
        width = max(self._underlay_canvas.winfo_width(), 1)
        height = max(self._underlay_canvas.winfo_height(), 1)
        self._underlay_canvas.create_text(
            width // 2,
            height // 2,
            anchor="center",
            fill="#dddddd",
            text=message,
            font=("TkDefaultFont", 10, "bold"),
        )

    def _render_underlay(self) -> None:
        self._underlay_refresh_after = None
        if self._underlay_canvas is None:
            return
        if self._underlay_data is None:
            self._show_underlay_message("Select an underlay scan.")
            return
        if not HAS_PIL:
            self._show_underlay_message("Install pillow to view underlay.")
            return
        plane = self._underlay_plane_var.get()
        slice_idx = int(self._underlay_slice_var.get())
        data = self._underlay_data
        if plane == "xz":
            slice_idx = max(0, min(slice_idx, data.shape[1] - 1))
            self._underlay_slice_var.set(slice_idx)
            img = data[:, slice_idx, :].T
            box = self._box_for_plane("xz", slice_idx)
        elif plane == "yz":
            slice_idx = max(0, min(slice_idx, data.shape[0] - 1))
            self._underlay_slice_var.set(slice_idx)
            img = data[slice_idx, :, :]
            box = self._box_for_plane("yz", slice_idx)
        else:
            slice_idx = max(0, min(slice_idx, data.shape[2] - 1))
            self._underlay_slice_var.set(slice_idx)
            img = data[:, :, slice_idx].T
            box = self._box_for_plane("xy", slice_idx)
        if np.iscomplexobj(img):
            img = np.abs(img)
        img = img.astype(float, copy=False)
        vmin, vmax = np.nanpercentile(img, (1, 99))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_display = np.flipud(img_uint8)

        pil_img = Image.fromarray(img_display, mode="L")
        canvas_w = max(self._underlay_canvas.winfo_width(), 1)
        canvas_h = max(self._underlay_canvas.winfo_height(), 1)
        aspect = pil_img.width / max(pil_img.height, 1)
        if self._underlay_res is not None:
            if plane == "xz":
                width_mm = img.shape[1] * float(self._underlay_res[0])
                height_mm = img.shape[0] * float(self._underlay_res[2])
            elif plane == "yz":
                width_mm = img.shape[1] * float(self._underlay_res[2])
                height_mm = img.shape[0] * float(self._underlay_res[1])
            else:
                width_mm = img.shape[1] * float(self._underlay_res[0])
                height_mm = img.shape[0] * float(self._underlay_res[1])
            if width_mm > 0 and height_mm > 0:
                aspect = width_mm / height_mm
        canvas_aspect = canvas_w / max(canvas_h, 1)
        if canvas_aspect >= aspect:
            target_h = canvas_h
            target_w = max(int(target_h * aspect), 1)
        else:
            target_w = canvas_w
            target_h = max(int(target_w / aspect), 1)

        resampling = getattr(Image, "Resampling", Image)
        resample = getattr(resampling, "NEAREST")
        pil_img = pil_img.resize((target_w, target_h), resample)
        self._underlay_image = ImageTk.PhotoImage(pil_img)
        x = (canvas_w - target_w) // 2
        y = (canvas_h - target_h) // 2
        self._underlay_canvas.delete("all")
        self._underlay_canvas.create_image(x, y, anchor="nw", image=self._underlay_image)
        self._draw_underlay_box(box, img.shape, (x, y, target_w, target_h))

        axis = self._plane_axis()
        size = int(self._underlay_data.shape[axis])
        self._underlay_slice_label.set(f"Slice: {slice_idx + 1}/{size}")

    def _resolve_voxel_box(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        voxel_pos, voxel_size = self.get_voxel_info()
        if voxel_pos is None or voxel_size is None or self._underlay_affine is None or self._underlay_res is None:
            return None
        pos = np.asarray(voxel_pos, dtype=float).reshape(-1)
        size = np.asarray(voxel_size, dtype=float).reshape(-1)
        if pos.size < 3 or size.size < 3:
            return None

        def _box_for_affine(affine: np.ndarray, res: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            try:
                inv_aff = np.linalg.inv(affine)
                center = inv_aff @ np.array([pos[0], pos[1], pos[2], 1.0], dtype=float)
                center_idx = center[:3]
                half = (size[:3] / res[:3]) / 2.0
                mins = np.floor(center_idx - half).astype(int)
                maxs = np.ceil(center_idx + half).astype(int)
                return mins, maxs
            except Exception:
                return None

        def _box_for_index() -> Optional[Tuple[np.ndarray, np.ndarray]]:
            if self._underlay_data is None:
                return None
            shape = np.asarray(self._underlay_data.shape[:3], dtype=float)
            if np.all(pos[:3] >= 0) and np.all(pos[:3] < shape):
                if np.all(size[:3] >= 0) and np.all(size[:3] <= shape):
                    half = size[:3] / 2.0
                    mins = np.floor(pos[:3] - half).astype(int)
                    maxs = np.ceil(pos[:3] + half).astype(int)
                    return mins, maxs
            return None

        index_box = _box_for_index()
        if index_box is not None:
            return index_box

        if self._underlay_affine is None or self._underlay_res is None or self._underlay_data is None:
            return None
        box = _box_for_affine(self._underlay_affine, self._underlay_res)
        if box is None:
            return None
        mins, maxs = box
        shape = np.asarray(self._underlay_data.shape[:3])
        if np.all(maxs >= 0) and np.all(mins < shape):
            return box
        return None

    def _orient_underlay(self, data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            ornt = nib.orientations.io_orientation(affine)
            target = np.array([[0, 1], [1, 1], [2, 1]])
            transform = nib.orientations.ornt_transform(ornt, target)
            data = nib.orientations.apply_orientation(data, transform)
            affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
            return data, affine
        except Exception:
            return data, affine

    def _box_for_plane(self, plane: str, slice_idx: int) -> Optional[Tuple[int, int, int, int]]:
        if self._underlay_box is None:
            return None
        mins, maxs = self._underlay_box
        x0, y0, z0 = mins.tolist()
        x1, y1, z1 = maxs.tolist()
        if plane == "xy":
            if slice_idx < z0 or slice_idx > z1:
                return None
            return y0, x0, y1, x1
        if plane == "xz":
            if slice_idx < y0 or slice_idx > y1:
                return None
            return z0, x0, z1, x1
        if plane == "yz":
            if slice_idx < x0 or slice_idx > x1:
                return None
            return y0, z0, y1, z1
        return None

    def _draw_underlay_box(
        self,
        box: Optional[Tuple[int, int, int, int]],
        shape: Tuple[int, int],
        layout: Tuple[int, int, int, int],
    ) -> None:
        if box is None or self._underlay_canvas is None:
            return
        row0, col0, row1, col1 = box
        rows, cols = shape
        row0 = max(0, min(row0, rows - 1))
        row1 = max(0, min(row1, rows - 1))
        col0 = max(0, min(col0, cols - 1))
        col1 = max(0, min(col1, cols - 1))
        row_min, row_max = sorted((row0, row1))
        col_min, col_max = sorted((col0, col1))
        x, y, target_w, target_h = layout
        x0 = x + col_min * target_w / cols
        x1 = x + (col_max + 1) * target_w / cols
        disp_row_min = rows - 1 - row_max
        disp_row_max = rows - 1 - row_min
        y0 = y + disp_row_min * target_h / rows
        y1 = y + (disp_row_max + 1) * target_h / rows
        self._underlay_canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#ff4444",
            width=2,
            dash=(3, 3),
            fill="#ff4444",
            stipple="gray50",
        )

    def _schedule_plot_refresh(self) -> None:
        if self._plot_refresh_after is not None:
            try:
                self.after_cancel(self._plot_refresh_after)
            except Exception:
                pass
            self._plot_refresh_after = None
        self._plot_refresh_after = self.after_idle(self._refresh_plot)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if HAS_MPL and self._spec_canvas is not None and self._spec_figure is not None:
            widget = self._spec_canvas.get_tk_widget()
            width = max(widget.winfo_width(), 1)
            height = max(widget.winfo_height(), 1)
            dpi = float(self._spec_figure.get_dpi() or 100.0)
            widget.configure(width=width, height=height)
            self._spec_figure.set_size_inches(width / dpi, height / dpi, forward=True)
            try:
                self._spec_canvas.draw()
            except Exception:
                pass
        if self._mrs_data is None:
            if self._last_message:
                self._render_message(self._last_message)
            return
        self._schedule_plot_refresh()

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        tip = _ToolTip(widget, text)
        self._tooltips.append(tip)
        widget.bind("<Enter>", tip.show, add="+")
        widget.bind("<Leave>", tip.hide, add="+")

    def _refresh_avg_dim_controls(self) -> None:
        if self._avg_dim_frame is None:
            return
        for child in self._avg_dim_frame.winfo_children():
            child.destroy()
        self._avg_dim_vars = {}
        if self._mrs_data is None:
            return
        data = np.asarray(self._mrs_data)
        if data.ndim < 4:
            ttk.Label(self._avg_dim_frame, text="Average dims: n/a").grid(row=0, column=0, sticky="w")
            return
        shape = data.reshape(data.shape[3:]).shape
        order = self._mrs_order or tuple(f"dim{idx}" for idx in range(len(shape)))
        col = 0
        shown = False
        for axis in range(1, len(shape)):
            label = order[axis] if axis < len(order) else f"dim{axis}"
            if shape[axis] <= 1:
                continue
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(
                self._avg_dim_frame,
                text=f"Average {label} (n={shape[axis]})",
                variable=var,
                command=self._schedule_plot_refresh,
            )
            chk.grid(row=0, column=col, sticky="w", padx=(0, 8))
            self._add_tooltip(chk, f"Toggle averaging over {label}.")
            self._avg_dim_vars[label] = var
            col += 1
            shown = True
        if not shown:
            summary = ", ".join(str(n) for n in shape[1:]) if len(shape) > 1 else "n/a"
            ttk.Label(self._avg_dim_frame, text=f"Average dims: n/a (sizes: {summary})").grid(
                row=0, column=0, sticky="w"
            )


class UnderlaySelector(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        app: Any,
        on_select,
        selected_scan_id: Optional[int] = None,
        selected_reco_id: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self.title("Select Underlay")
        self.geometry("520x360")
        self.minsize(460, 320)
        self._app = app
        self._on_select = on_select
        self._selected_scan_id = selected_scan_id
        self._selected_reco_id = selected_reco_id
        self._scan_ids: list[int] = []
        self._scan_info: Dict[int, Dict[str, Any]] = {}
        self._scan_list: Optional[tk.Listbox] = None
        self._reco_list: Optional[tk.Listbox] = None
        self._build_ui()
        self._load_scans()

    def _build_ui(self) -> None:
        body = ttk.Frame(self, padding=(10, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Scans").grid(row=0, column=0, sticky="w")
        ttk.Label(body, text="Recos").grid(row=0, column=1, sticky="w")

        scan_box = ttk.Frame(body)
        scan_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        scan_box.columnconfigure(0, weight=1)
        scan_box.rowconfigure(0, weight=1)
        self._scan_list = tk.Listbox(scan_box, height=12, exportselection=False)
        self._scan_list.grid(row=0, column=0, sticky="nsew")
        self._scan_list.bind("<<ListboxSelect>>", lambda _e: self._on_scan_select())
        scan_scroll = ttk.Scrollbar(scan_box, orient="vertical", command=self._scan_list.yview)
        scan_scroll.grid(row=0, column=1, sticky="ns")
        self._scan_list.configure(yscrollcommand=scan_scroll.set)

        reco_box = ttk.Frame(body)
        reco_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        reco_box.columnconfigure(0, weight=1)
        reco_box.rowconfigure(0, weight=1)
        self._reco_list = tk.Listbox(reco_box, height=12, exportselection=False)
        self._reco_list.grid(row=0, column=0, sticky="nsew")
        reco_scroll = ttk.Scrollbar(reco_box, orient="vertical", command=self._reco_list.yview)
        reco_scroll.grid(row=0, column=1, sticky="ns")
        self._reco_list.configure(yscrollcommand=reco_scroll.set)

        btns = ttk.Frame(body)
        btns.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        ttk.Button(btns, text="Select", command=self._select).grid(row=0, column=0, sticky="e")

    def _load_scans(self) -> None:
        study = getattr(self._app, "_study", None)
        info_full = getattr(self._app, "_info_full", {}) or {}
        scan_info_all = info_full.get("Scan(s)", {}) if isinstance(info_full, dict) else {}
        if isinstance(scan_info_all, dict):
            self._scan_info = scan_info_all
        if study is None:
            return
        self._scan_ids = list(study.avail.keys())
        if self._scan_list is None:
            return
        self._scan_list.delete(0, tk.END)
        for scan_id in self._scan_ids:
            info = self._scan_info.get(scan_id, {})
            protocol = info.get("Protocol", "N/A")
            self._scan_list.insert(tk.END, f"{scan_id:03d} :: {protocol}")
        if self._scan_ids:
            select_idx = 0
            if self._selected_scan_id in self._scan_ids:
                select_idx = self._scan_ids.index(self._selected_scan_id)
            self._scan_list.selection_set(select_idx)
            self._on_scan_select()

    def _on_scan_select(self) -> None:
        if self._scan_list is None or self._reco_list is None:
            return
        selection = self._scan_list.curselection()
        if not selection:
            return
        scan_id = self._scan_ids[int(selection[0])]
        info = self._scan_info.get(scan_id, {}) if isinstance(self._scan_info, dict) else {}
        recos = info.get("Reco(s)", {}) if isinstance(info, dict) else {}
        study = getattr(self._app, "_study", None)
        scan = study.avail.get(scan_id) if study is not None else None
        self._reco_list.delete(0, tk.END)
        reco_ids = list(scan.avail.keys()) if scan is not None else []
        for reco_id in reco_ids:
            reco_info = recos.get(reco_id, {}) if isinstance(recos, dict) else {}
            label = reco_info.get("Type", "N/A")
            self._reco_list.insert(tk.END, f"{int(reco_id):03d} :: {label}")
        if reco_ids:
            select_idx = 0
            if self._selected_reco_id in reco_ids:
                select_idx = reco_ids.index(self._selected_reco_id)
            self._reco_list.selection_set(select_idx)

    def _select(self) -> None:
        if self._scan_list is None or self._reco_list is None:
            return
        scan_sel = self._scan_list.curselection()
        reco_sel = self._reco_list.curselection()
        if not scan_sel:
            return
        if not reco_sel and self._reco_list.size() > 0:
            reco_sel = (0,)
            self._reco_list.selection_set(0)
        if not reco_sel:
            return
        scan_id = self._scan_ids[int(scan_sel[0])]
        reco_text = self._reco_list.get(reco_sel[0])
        try:
            reco_id = int(reco_text.split("::")[0].strip())
        except Exception:
            return
        study = getattr(self._app, "_study", None)
        if study is None:
            return
        scan = study.avail.get(scan_id)
        label = f"{scan_id:03d} :: {reco_text}"
        self._on_select(label, scan, reco_id)
        self.destroy()


class _ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip: Optional[tk.Toplevel] = None

    def show(self, _event: Optional[tk.Event] = None) -> None:
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 8
        tip = tk.Toplevel(self._widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tip, text=self._text, background="#fff4c2", relief="solid", borderwidth=1)
        label.pack(ipadx=6, ipady=3)
        self._tip = tip

    def hide(self, _event: Optional[tk.Event] = None) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None
