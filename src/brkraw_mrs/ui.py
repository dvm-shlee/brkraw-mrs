from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk

from .hook import _load_mrs_data

logger = logging.getLogger("brkraw.mrs")

MAX_PLOT_POINTS = 2048
MAX_PLOT_TRACES = 16


class MRSPanel(ttk.Frame):
    def __init__(self, parent: tk.Misc, *, app: Any) -> None:
        super().__init__(parent)
        self._app = app
        self._mrs_data: Optional[np.ndarray] = None
        self._mrs_order: Optional[Tuple[str, ...]] = None
        self._mrs_meta: Dict[str, Any] = {}

        self._status_var = tk.StringVar(value="")
        self._lb_var = tk.DoubleVar(value=0.0)
        self._phase_var = tk.DoubleVar(value=0.0)
        self._shift_var = tk.DoubleVar(value=0.0)
        self._lb_min_var = tk.DoubleVar(value=0.0)
        self._lb_max_var = tk.DoubleVar(value=10.0)
        self._phase_min_var = tk.DoubleVar(value=-180.0)
        self._phase_max_var = tk.DoubleVar(value=180.0)
        self._shift_min_var = tk.DoubleVar(value=-20.0)
        self._shift_max_var = tk.DoubleVar(value=20.0)

        self._plot_domain_var = tk.StringVar(value="Spectrum")
        self._x_unit_var = tk.StringVar(value="ppm")
        self._standard_mrs_var = tk.BooleanVar(value=True)
        self._spectrum_view_var = tk.StringVar(value="Real")
        self._fid_view_var = tk.StringVar(value="Real")
        self._normalize_var = tk.BooleanVar(value=False)

        self._avg_dim_vars: Dict[str, tk.BooleanVar] = {}
        self._avg_dim_frame: Optional[ttk.Frame] = None

        self._spec_canvas: Optional[tk.Canvas] = None
        self._last_message: Optional[str] = None
        self._plot_refresh_after: Optional[str] = None
        self._plot_domain_combo: Optional[ttk.Combobox] = None
        self._x_axis_combo: Optional[ttk.Combobox] = None
        self._spectrum_view_combo: Optional[ttk.Combobox] = None
        self._fid_view_combo: Optional[ttk.Combobox] = None
        self._standard_plot_check: Optional[ttk.Checkbutton] = None

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
        body.rowconfigure(0, weight=1)

        spectrum = ttk.Frame(body)
        spectrum.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        spectrum.columnconfigure(0, weight=1)
        spectrum.rowconfigure(0, weight=1)

        controls_row = 1
        self._spec_canvas = tk.Canvas(spectrum, background="#101010", highlightthickness=0)
        self._spec_canvas.grid(row=0, column=0, sticky="nsew")
        self._spec_canvas.bind("<Configure>", self._on_canvas_resize)

        controls = ttk.Frame(spectrum)
        controls.grid(row=controls_row, column=0, sticky="ew", pady=(6, 0))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Plot & preprocessing").grid(row=0, column=0, columnspan=2, sticky="w")

        self._avg_dim_frame = ttk.Frame(controls)
        self._avg_dim_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        plot_opts = ttk.Frame(controls)
        plot_opts.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        plot_opts.columnconfigure(1, weight=1)
        plot_opts.columnconfigure(3, weight=1)

        ttk.Label(plot_opts, text="View").grid(row=0, column=0, sticky="w")
        domain_combo = ttk.Combobox(
            plot_opts,
            state="readonly",
            values=["Spectrum", "FID"],
            width=10,
            textvariable=self._plot_domain_var,
        )
        domain_combo.grid(row=0, column=1, sticky="ew", padx=(6, 10))
        self._plot_domain_combo = domain_combo
        self._add_tooltip(domain_combo, "Choose to plot frequency-domain spectrum or time-domain FID.")
        domain_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_plot_controls())

        ttk.Label(plot_opts, text="X-axis").grid(row=0, column=2, sticky="w")
        x_combo = ttk.Combobox(
            plot_opts,
            state="readonly",
            values=["Hz", "ppm"],
            width=8,
            textvariable=self._x_unit_var,
        )
        x_combo.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        self._x_axis_combo = x_combo
        self._add_tooltip(x_combo, "Frequency axis unit for spectrum plots.")
        x_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_plot_controls())

        view_opts = ttk.Frame(controls)
        view_opts.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        view_opts.columnconfigure(1, weight=1)
        view_opts.columnconfigure(3, weight=1)

        standard_check = ttk.Checkbutton(
            view_opts,
            text="X: ppm (reversed)",
            variable=self._standard_mrs_var,
            command=self._sync_plot_controls,
        )
        standard_check.grid(row=0, column=0, columnspan=2, sticky="w")
        self._standard_plot_check = standard_check
        ttk.Checkbutton(
            view_opts,
            text="Y: normalize (max=1)",
            variable=self._normalize_var,
            command=self._schedule_plot_refresh,
        ).grid(row=0, column=2, columnspan=2, sticky="w")

        ttk.Label(view_opts, text="Spectrum").grid(row=1, column=0, sticky="w")
        spec_combo = ttk.Combobox(
            view_opts,
            state="readonly",
            values=["Real", "Magnitude"],
            width=10,
            textvariable=self._spectrum_view_var,
        )
        spec_combo.grid(row=1, column=1, sticky="ew", padx=(6, 10))
        self._spectrum_view_combo = spec_combo
        self._add_tooltip(spec_combo, "Choose real or magnitude spectrum.")
        spec_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_plot_refresh())

        ttk.Label(view_opts, text="FID").grid(row=1, column=2, sticky="w")
        fid_combo = ttk.Combobox(
            view_opts,
            state="readonly",
            values=["Real", "Magnitude"],
            width=10,
            textvariable=self._fid_view_var,
        )
        fid_combo.grid(row=1, column=3, sticky="ew", padx=(6, 0))
        self._fid_view_combo = fid_combo
        self._add_tooltip(fid_combo, "Choose real or magnitude FID.")
        fid_combo.bind("<<ComboboxSelected>>", lambda _e: self._schedule_plot_refresh())

        self._sync_plot_controls()

        self._add_range_control(
            controls,
            row=4,
            label="Line broadening (Hz)",
            value_var=self._lb_var,
            min_var=self._lb_min_var,
            max_var=self._lb_max_var,
            tooltip="Apply exponential line broadening in time domain (Hz).",
        )
        self._add_range_control(
            controls,
            row=5,
            label="Phase0 (deg)",
            value_var=self._phase_var,
            min_var=self._phase_min_var,
            max_var=self._phase_max_var,
            tooltip="Zero-order phase correction in degrees.",
        )
        self._add_range_control(
            controls,
            row=6,
            label="Freq shift (Hz)",
            value_var=self._shift_var,
            min_var=self._shift_min_var,
            max_var=self._shift_max_var,
            tooltip="Frequency shift applied to the spectrum (Hz).",
        )

        actions = ttk.Frame(controls)
        actions.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Reset", command=self._reset_controls).grid(row=0, column=0, sticky="e")

        status = ttk.Label(self, textvariable=self._status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def refresh_from_viewer(self) -> None:
        scan = getattr(self._app, "_scan", None)
        if scan is None:
            self._mrs_data = None
            self._mrs_order = None
            self._mrs_meta = {}
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

    def _refresh_plot(self) -> None:
        self._plot_refresh_after = None
        if self._mrs_data is None or self._mrs_order is None:
            self._show_message("No MRS data.")
            return
        fid, dwell = self._process_fid()
        if fid is None or dwell is None:
            self._show_message("Failed to process FID.")
            return
        plot_domain = (self._plot_domain_var.get() or "Spectrum").lower()
        status = f"Points: {fid.shape[0]}  Dwell: {dwell:.6f}s"
        status_extra = self._build_status_meta()
        if status_extra:
            status += f"  {status_extra}"

        if plot_domain == "fid":
            series, time_axis = self._build_fid_series(fid, dwell)
            mode = (self._fid_view_var.get() or "Real").lower()
            series = self._select_component(series, mode)
            x_label = "Time (ms)"
            title = f"FID ({mode.title()})"
            self._plot_lines(
                time_axis * 1000.0,
                series,
                x_label=x_label,
                y_label="Normalized signal" if self._normalize_var.get() else "Signal",
                title=title,
                invert_x=False,
                normalize=bool(self._normalize_var.get()),
            )
        else:
            spectra, freq = self._build_spectra(fid, dwell)
            mode = (self._spectrum_view_var.get() or "Real").lower()
            spectra = self._select_component(spectra, mode)
            standard = bool(self._standard_mrs_var.get())
            x_axis, x_label, invert, note = self._resolve_spectrum_axis(freq, standard=standard)
            if note:
                status += f"  {note}"
            title = f"Spectrum ({mode.title()})"
            self._plot_lines(
                x_axis,
                spectra,
                x_label=x_label,
                y_label="Normalized signal" if self._normalize_var.get() else "Amplitude",
                title=title,
                invert_x=invert,
                normalize=bool(self._normalize_var.get()),
            )
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

        if fid.ndim == 0:
            return None, None
        fid = np.atleast_1d(fid)

        dwell = float(self._mrs_meta.get("DwellTime") or 0.0)
        if dwell == 0.0:
            dwell = 1.0

        t = np.arange(int(fid.shape[0])) * dwell
        lb = self._safe_var_float(self._lb_var, 0.0)
        if lb:
            fid = fid * np.exp(-lb * np.pi * t)
        phase = np.deg2rad(self._safe_var_float(self._phase_var, 0.0))
        if phase:
            fid = fid * np.exp(1j * phase)
        shift = self._safe_var_float(self._shift_var, 0.0)
        if shift:
            fid = fid * np.exp(1j * 2 * np.pi * shift * t)

        return fid, dwell

    def _build_spectra(self, fid: np.ndarray, dwell: float) -> Tuple[list[np.ndarray], np.ndarray]:
        data = np.asarray(fid)
        if data.ndim == 1:
            spectra = [np.fft.fftshift(np.fft.fft(data))]
        else:
            spectra = []
            dims = tuple(int(dim) for dim in data.shape[1:])
            for idx in product(*[range(dim) for dim in dims]):
                series = data[(slice(None),) + idx]
                spectra.append(np.fft.fftshift(np.fft.fft(series)))
        freq = np.fft.fftshift(np.fft.fftfreq(data.shape[0], d=dwell))
        return spectra, freq

    def _build_fid_series(self, fid: np.ndarray, dwell: float) -> Tuple[list[np.ndarray], np.ndarray]:
        data = np.asarray(fid)
        if data.ndim == 1:
            series = [data]
        else:
            series = []
            dims = tuple(int(dim) for dim in data.shape[1:])
            for idx in product(*[range(dim) for dim in dims]):
                trace = data[(slice(None),) + idx]
                series.append(trace)
        time_axis = np.arange(data.shape[0]) * dwell
        return series, time_axis

    def _plot_lines(
        self,
        x_axis: np.ndarray,
        series_list: list[np.ndarray],
        *,
        x_label: str,
        y_label: str,
        title: str,
        invert_x: bool,
        normalize: bool,
    ) -> None:
        canvas = self._spec_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if x_axis.size == 0 or not series_list:
            self._render_message("No plot data.")
            return

        if len(series_list) > MAX_PLOT_TRACES:
            idx = np.linspace(0, len(series_list) - 1, MAX_PLOT_TRACES, dtype=int)
            series_list = [series_list[i] for i in idx]

        processed_series: list[np.ndarray] = []
        max_abs = 0.0
        for series in series_list:
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                continue
            if normalize:
                denom = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 0.0
                if denom > 0:
                    arr = arr / denom
            processed_series.append(arr)
            try:
                max_abs = max(max_abs, float(np.nanmax(np.abs(arr))))
            except Exception:
                pass
        if not processed_series:
            self._render_message("No plot data.")
            return

        scale_exp = 0
        scale_factor = 1.0
        scale_label: Optional[str] = None
        if not normalize and max_abs > 0.0:
            scale_exp = int(np.floor(np.log10(max_abs)))
            if abs(scale_exp) >= 3:
                scale_factor = 10.0 ** scale_exp
                scale_label = f"x1e{scale_exp}"
        if scale_factor != 1.0:
            processed_series = [arr / scale_factor for arr in processed_series]

        x_min = float(np.nanmin(x_axis))
        x_max = float(np.nanmax(x_axis))
        if np.isclose(x_min, x_max):
            x_max = x_min + 1.0
        y_min = float("inf")
        y_max = float("-inf")
        for series in processed_series:
            if series.size == 0:
                continue
            y_min = min(y_min, float(np.nanmin(series)))
            y_max = max(y_max, float(np.nanmax(series)))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            self._render_message("No plot data.")
            return
        if np.isclose(y_min, y_max):
            y_max = y_min + 1.0

        pad_left = 78
        pad_right = 24
        pad_top = 38
        pad_bottom = 64
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom
        if plot_w <= 10 or plot_h <= 10:
            return

        x0 = pad_left
        y0 = pad_top
        x1 = pad_left + plot_w
        y1 = pad_top + plot_h
        canvas.create_rectangle(x0, y0, x1, y1, outline="#2f2f2f", fill="#101010")

        scale_x = plot_w / (x_max - x_min)
        scale_y = plot_h / (y_max - y_min)

        def _to_canvas(x_val: float, y_val: float) -> tuple[float, float]:
            if invert_x:
                x_pos = x0 + (x_max - x_val) * scale_x
            else:
                x_pos = x0 + (x_val - x_min) * scale_x
            return x_pos, y0 + (y_max - y_val) * scale_y

        ticks = 4
        tick_label_y = y1 + 8
        for i in range(ticks + 1):
            tx = x0 + i * plot_w / ticks
            if invert_x:
                x_val = x_max - i * (x_max - x_min) / ticks
            else:
                x_val = x_min + i * (x_max - x_min) / ticks
            canvas.create_line(tx, y1, tx, y1 + 4, fill="#777777")
            canvas.create_text(tx, tick_label_y, text=self._format_tick(x_val), anchor="n", fill="#cccccc")
        for i in range(ticks + 1):
            ty = y0 + i * plot_h / ticks
            y_val = y_max - i * (y_max - y_min) / ticks
            canvas.create_line(x0 - 4, ty, x0, ty, fill="#777777")
            canvas.create_text(x0 - 6, ty, text=self._format_tick(y_val), anchor="e", fill="#cccccc")

        if y_min < 0 < y_max:
            _, zero_y = _to_canvas(x_min, 0.0)
            canvas.create_line(x0, zero_y, x1, zero_y, fill="#444444")

        max_points = min(max(int(plot_w * 2), 32), MAX_PLOT_POINTS)
        if x_axis.size > max_points:
            step = max(1, int(np.ceil(x_axis.size / max_points)))
            idx = slice(0, None, step)
            x_plot = x_axis[idx]
        else:
            x_plot = x_axis
            idx = slice(None)

        for series in processed_series:
            if series.size == 0:
                continue
            series = series[idx]
            coords: list[float] = []
            for xv, yv in zip(x_plot, series):
                cx, cy = _to_canvas(float(xv), float(yv))
                coords.extend([cx, cy])
            if len(coords) >= 4:
                canvas.create_line(coords, fill="#66ccff", width=1)

        title_y = max(10, y0 - 18)
        x_label_y = min(height - 10, y1 + 26)
        canvas.create_text((x0 + x1) / 2, title_y, text=title, fill="#dddddd")
        canvas.create_text((x0 + x1) / 2, x_label_y, text=x_label, fill="#cccccc")
        canvas.create_text(12, y0 + 4, text=y_label, anchor="nw", fill="#cccccc")
        if scale_label:
            canvas.create_text(x0 - 6, y0 - 12, text=scale_label, anchor="w", fill="#999999")

    def _select_component(self, series_list: list[np.ndarray], mode: str) -> list[np.ndarray]:
        if mode.startswith("mag"):
            return [np.abs(series) for series in series_list]
        return [np.real(series) for series in series_list]

    def _resolve_spectrum_axis(
        self,
        freq_hz: np.ndarray,
        *,
        standard: bool,
    ) -> Tuple[np.ndarray, str, bool, Optional[str]]:
        use_ppm = self._x_unit_var.get() == "ppm"
        if use_ppm:
            sf_mhz = self._get_meta_float("SpectrometerFrequency")
            if sf_mhz is None or sf_mhz == 0.0:
                return freq_hz, "Hz", False, "ppm axis unavailable (missing spectrometer frequency)."
            ref_ppm = self._get_meta_float("TxOffset") or 0.0
            ppm = freq_hz / (sf_mhz * 1e6) + ref_ppm
            return ppm, "ppm", standard, None
        return freq_hz, "Hz", standard, None

    def _build_status_meta(self) -> str:
        parts: list[str] = []
        sf = self._get_meta_float("SpectrometerFrequency")
        if sf is not None:
            parts.append(f"SF: {sf:.3f} MHz")
        sw = self._get_meta_float("SpectralWidth")
        if sw is not None:
            parts.append(f"SW: {sw:.1f} Hz")
        nucleus = self._mrs_meta.get("ResonantNucleus")
        if isinstance(nucleus, (list, tuple)) and nucleus:
            parts.append(f"Nucleus: {nucleus[0]}")
        elif isinstance(nucleus, str):
            parts.append(f"Nucleus: {nucleus}")
        return "  ".join(parts)

    def _get_meta_float(self, key: str) -> Optional[float]:
        value = self._mrs_meta.get(key)
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size:
                value = value.flat[0]
            else:
                return None
        elif isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _sync_plot_controls(self) -> None:
        domain = (self._plot_domain_var.get() or "Spectrum").lower()
        x_is_ppm = self._x_unit_var.get() == "ppm"
        if self._x_axis_combo is not None:
            self._x_axis_combo.configure(state="readonly" if domain == "spectrum" else "disabled")
        if self._standard_plot_check is not None:
            if domain != "spectrum":
                self._standard_plot_check.configure(state="disabled")
                self._standard_mrs_var.set(False)
            elif x_is_ppm:
                self._standard_plot_check.configure(state="normal")
            else:
                self._standard_plot_check.configure(state="disabled")
                self._standard_mrs_var.set(False)
        if self._spectrum_view_combo is not None:
            self._spectrum_view_combo.configure(state="readonly" if domain == "spectrum" else "disabled")
        if self._fid_view_combo is not None:
            self._fid_view_combo.configure(state="readonly" if domain == "fid" else "disabled")
        self._schedule_plot_refresh()

    def _reset_controls(self) -> None:
        self._lb_var.set(0.0)
        self._phase_var.set(0.0)
        self._shift_var.set(0.0)
        self._lb_min_var.set(0.0)
        self._lb_max_var.set(10.0)
        self._phase_min_var.set(-180.0)
        self._phase_max_var.set(180.0)
        self._shift_min_var.set(-20.0)
        self._shift_max_var.set(20.0)
        self._plot_domain_var.set("Spectrum")
        self._x_unit_var.set("ppm")
        self._standard_mrs_var.set(True)
        self._spectrum_view_var.set("Real")
        self._fid_view_var.set("Real")
        self._normalize_var.set(False)
        self._sync_plot_controls()

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
        canvas = self._spec_canvas
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

    def _schedule_plot_refresh(self) -> None:
        if self._plot_refresh_after is not None:
            try:
                self.after_cancel(self._plot_refresh_after)
            except Exception:
                pass
            self._plot_refresh_after = None
        self._plot_refresh_after = self.after_idle(self._refresh_plot)

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._mrs_data is None:
            if self._last_message:
                self._render_message(self._last_message)
            return
        self._schedule_plot_refresh()

    def _format_tick(self, value: float) -> str:
        abs_val = abs(value)
        if abs_val >= 1000:
            return f"{value:.0f}"
        if abs_val >= 10:
            return f"{value:.1f}"
        if abs_val >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def _safe_float(self, value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _safe_var_float(self, var: tk.Variable, fallback: float) -> float:
        try:
            value = var.get()
        except tk.TclError:
            return fallback
        return self._safe_float(value, fallback)

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        tip = _ToolTip(widget, text)
        self._tooltips.append(tip)
        widget.bind("<Enter>", tip.show, add="+")
        widget.bind("<Leave>", tip.hide, add="+")

    def _add_range_control(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        value_var: tk.DoubleVar,
        min_var: tk.DoubleVar,
        max_var: tk.DoubleVar,
        tooltip: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=(6, 0))
        frame.columnconfigure(3, weight=1)

        value_entry = ttk.Entry(frame, textvariable=value_var, width=8)
        value_entry.grid(row=0, column=0, sticky="w")
        min_entry = ttk.Entry(frame, textvariable=min_var, width=6)
        min_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))
        scale = ttk.Scale(frame, variable=value_var, orient=tk.HORIZONTAL)
        scale.grid(row=0, column=2, columnspan=2, sticky="ew", padx=(6, 6))
        max_entry = ttk.Entry(frame, textvariable=max_var, width=6)
        max_entry.grid(row=0, column=4, sticky="e")

        self._add_tooltip(value_entry, tooltip)
        self._add_tooltip(min_entry, "Minimum value for slider range.")
        self._add_tooltip(max_entry, "Maximum value for slider range.")

        def _update_scale(*_args: object) -> None:
            min_val = self._safe_var_float(min_var, 0.0)
            max_val = self._safe_var_float(max_var, 0.0)
            if max_val <= min_val:
                max_val = min_val + 1.0
                max_var.set(max_val)
            scale.configure(from_=min_val, to=max_val)
            current = self._safe_var_float(value_var, min_val)
            if current < min_val:
                value_var.set(min_val)
            elif current > max_val:
                value_var.set(max_val)

        min_var.trace_add("write", _update_scale)
        max_var.trace_add("write", _update_scale)
        value_var.trace_add("write", lambda *_args: self._schedule_plot_refresh())
        scale.configure(command=lambda _v: self._schedule_plot_refresh())
        _update_scale()

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
        shape = tuple(int(dim) for dim in data.reshape(data.shape[3:]).shape)
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
