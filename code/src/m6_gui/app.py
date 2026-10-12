"""M6 — Researcher-facing GUI (FR-6.1-FR-6.4).

Tkinter (stdlib) + Pillow, so the app has no ML/GPU dependency of its own; it drives M1-M5
through :mod:`src.m6_gui.orchestrator` on a background thread so the window never freezes.

Layout (light theme, resizable): a header with the scan and output folder, a scrollable
sidebar with the one-click pipeline and one card per module, and a viewer with
Viewer / Review / Log tabs. Every module card has its own input file field(s), so any
module can be run on its own; the cards fill each other's inputs automatically when run in
sequence. What gets drawn (panels, bounding boxes, hover read-outs) is decided in
:mod:`src.m6_gui.viewmodel`.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import numpy as np
from PIL import Image, ImageTk

from src.imaging import to_display_image
from src.m6_gui import viewmodel as vm
from src.m6_gui.orchestrator import PipelineOrchestrator

# Kept for callers/tests that convert a single array into a Tk image.
PREVIEW_SIZE = (440, 440)

# ---- light theme ---------------------------------------------------------------------------
BG = "#f4f6f9"
SURFACE = "#ffffff"
BORDER = "#d9dee5"
TEXT = "#1f2933"
MUTED = "#667383"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
ACCENT_SOFT = "#e8effd"
OK = "#16a34a"
WARN = "#92400e"
WARN_BG = "#fef3c7"
ERR = "#dc2626"
ERR_BG = "#fee2e2"
FONT = "Segoe UI"

STATUS_GLYPH = {"idle": ("○", MUTED), "running": ("◐", ACCENT), "done": ("●", OK), "failed": ("✕", ERR)}

VIEWS = {"Scan slice": "scan", "Segmentation": "segmentation", "Flattened page": "flattened", "Ink detection": "ink"}
PHASE_VIEW = {"m1": "Scan slice", "m2": "Segmentation", "m3": "Flattened page", "spiral_fitting": "Flattened page", "m4": "Ink detection"}

H5_TYPES = [("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]


def array_to_photoimage(arr: np.ndarray, size: tuple[int, int] = PREVIEW_SIZE, cmap: str | None = None) -> ImageTk.PhotoImage:
    """Percentile-stretched, LANCZOS-resampled preview of one array as a Tk image."""
    return ImageTk.PhotoImage(to_display_image(arr, size=size, cmap=cmap))


def labels_to_photoimage(labels: np.ndarray, size: tuple[int, int] = PREVIEW_SIZE) -> ImageTk.PhotoImage:
    """Colourised segment-label preview as a Tk image."""
    return ImageTk.PhotoImage(to_display_image(labels, size=size, is_labels=True))


def apply_theme(root: tk.Misc) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=BG)
    style.configure(".", font=(FONT, 10), background=BG, foreground=TEXT, bordercolor=BORDER, focuscolor=BG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=MUTED, font=(FONT, 9))
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED, font=(FONT, 9))
    style.configure("CardTitle.TLabel", background=SURFACE, font=(FONT, 10, "bold"))
    style.configure("Card.TCheckbutton", background=SURFACE)
    style.configure("Card.TRadiobutton", background=SURFACE)
    style.map("Card.TCheckbutton", background=[("active", SURFACE)])
    style.map("Card.TRadiobutton", background=[("active", SURFACE)])
    style.configure("TButton", padding=(10, 5), background="#eef1f5", bordercolor=BORDER, lightcolor="#eef1f5", darkcolor="#eef1f5", borderwidth=1, relief="flat")
    style.map("TButton", background=[("active", ACCENT_SOFT), ("disabled", BG)], foreground=[("disabled", "#a0a9b5")])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, padding=(12, 6))
    style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#9db8f0")], foreground=[("disabled", "#ffffff")])
    style.configure("TScrollbar", background="#c9d0d9", troughcolor=BG, bordercolor=BG, arrowcolor=MUTED, lightcolor="#c9d0d9", darkcolor="#c9d0d9", relief="flat")
    style.map("TScrollbar", background=[("active", "#aab3bf")])
    style.configure("Small.TButton", padding=(6, 3), font=(FONT, 9))
    style.configure("TEntry", fieldbackground=SURFACE, bordercolor=BORDER, padding=4)
    style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE, bordercolor=BORDER, padding=3)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)])
    style.configure("TSpinbox", fieldbackground=SURFACE, bordercolor=BORDER, padding=3)
    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", padding=(16, 7), background=BG, foreground=MUTED)
    style.map("TNotebook.Tab", background=[("selected", SURFACE)], foreground=[("selected", ACCENT)])
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BORDER, bordercolor=BORDER, thickness=6)
    style.configure("Horizontal.TScale", background=SURFACE, troughcolor=BORDER)
    style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, rowheight=24, bordercolor=BORDER)
    style.configure("Treeview.Heading", background=BG, foreground=MUTED, font=(FONT, 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT_SOFT)], foreground=[("selected", TEXT)])


class WrapLabel(ttk.Label):
    """A label whose text re-wraps to its own width, so long messages never overflow or get clipped."""

    def __init__(self, master, pad: int = 4, **kw):
        super().__init__(master, **kw)
        self.bind("<Configure>", lambda e: self.configure(wraplength=max(60, e.width - pad)))


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container; the inner frame always matches the visible width (no horizontal overflow)."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self._sync())
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        self._sync()

    def _sync(self) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def _wheel(self, event) -> None:
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ModuleCard(tk.Frame):
    """One pipeline module: status, description, its own input files and options, and a Run button."""

    def __init__(self, master, title: str, description: str, on_run, run_label: str = "Run"):
        super().__init__(master, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        self.columnconfigure(0, weight=1)
        head = ttk.Frame(self, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.status = ttk.Label(head, text="○", style="Card.TLabel", foreground=MUTED)
        self.status.grid(row=0, column=1, sticky="e")
        WrapLabel(self, text=description, style="CardMuted.TLabel").grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 4))
        self.body = ttk.Frame(self, style="Card.TFrame")
        self.body.grid(row=2, column=0, sticky="ew", padx=10)
        self.body.columnconfigure(0, weight=1)
        self.error = WrapLabel(self, text="", style="Card.TLabel", foreground=ERR)
        self.error.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 0))
        self.error.grid_remove()
        self.button = ttk.Button(self, text=run_label, command=on_run)
        self.button.grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 10))
        self._row = 0

    def add_input(self, label: str, var: tk.StringVar, on_browse) -> None:
        ttk.Label(self.body, text=label, style="CardMuted.TLabel").grid(row=self._row, column=0, sticky="w", pady=(4, 0))
        row = ttk.Frame(self.body, style="Card.TFrame")
        row.grid(row=self._row + 1, column=0, sticky="ew")
        row.columnconfigure(0, weight=1)
        entry = ttk.Entry(row, textvariable=var)
        entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="Browse…", style="Small.TButton", command=on_browse).grid(row=0, column=1, padx=(4, 0))
        var.trace_add("write", lambda *_: entry.after_idle(lambda: entry.xview_moveto(1.0)))  # show the file name, not the folder
        self._row += 2

    def add_option(self, label: str, widget_factory) -> ttk.Widget:
        row = ttk.Frame(self.body, style="Card.TFrame")
        row.grid(row=self._row, column=0, sticky="ew", pady=(4, 0))
        row.columnconfigure(0, weight=1)
        WrapLabel(row, text=label, style="Card.TLabel").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        widget = widget_factory(row)
        widget.grid(row=0, column=1, sticky="e")
        self._row += 1
        return widget

    def set_status(self, state: str) -> None:
        glyph, colour = STATUS_GLYPH[state]
        self.status.configure(text=glyph, foreground=colour)

    def show_error(self, message: str | None) -> None:
        if message:
            self.error.configure(text=message)
            self.error.grid()
        else:
            self.error.grid_remove()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        try:  # crisp text on high-DPI Windows displays
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
        self.title("Virtual Unwrap — Researcher Console")
        self.geometry("1280x800")
        self.minsize(880, 560)
        apply_theme(self)

        self.orchestrator: PipelineOrchestrator | None = None
        self._ui_queue: queue.Queue = queue.Queue()
        self._busy = False
        self._view: vm.View | None = None
        self._placed: list[vm.Placed] = []
        self._selected: tuple[int, vm.Box] | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._base_images: dict[int, Image.Image] = {}
        self._request_token = 0
        self._redraw_job = None
        self._load_job = None
        self._cards: dict[str, ModuleCard] = {}
        self._auto_values: dict[str, str] = {}  # keyed by Tk variable name (variables themselves are unhashable)
        self._pipeline_notice: str | None = None

        self.v_scan = tk.StringVar()
        self.v_corrected = tk.StringVar()
        self.v_labels = tk.StringVar()
        self.v_flat = tk.StringVar()
        self.v_ink = tk.StringVar()
        self.v_m4_scan = tk.StringVar()
        self.v_out = tk.StringVar()
        self.engine_var = tk.StringVar(value="classical")
        self.skip_m1 = tk.BooleanVar(value=False)
        self.show_boxes = tk.BooleanVar(value=True)
        self.view_var = tk.StringVar(value="Scan slice")
        self.layer_var = tk.StringVar()
        self.z_var = tk.IntVar(value=0)

        self._build()
        self.after(100, self._poll_ui_queue)

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header()

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=(8, 6))
        side = ScrollFrame(paned)
        side.canvas.configure(width=340)
        paned.add(side, weight=0)
        self._build_sidebar(side.inner)
        main = ttk.Frame(paned)
        paned.add(main, weight=1)
        self._build_main(main)

        status = ttk.Frame(self)
        status.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        status.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Open a scan to begin.")
        WrapLabel(status, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=140)
        self.progress.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.progress.grid_remove()

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(2, weight=1)
        ttk.Label(bar, text="Virtual Unwrap", style="Card.TLabel", font=(FONT, 13, "bold")).grid(row=0, column=0, padx=(14, 14), pady=10)
        self.open_btn = ttk.Button(bar, text="Open scan…", style="Accent.TButton", command=self._on_open_scan)
        self.open_btn.grid(row=0, column=1, pady=10)
        self.scan_label = ttk.Label(bar, text="No scan loaded", style="CardMuted.TLabel", anchor="w")
        self.scan_label.grid(row=0, column=2, sticky="ew", padx=12)
        self.out_btn = ttk.Button(bar, text="Output folder…", command=self._on_choose_output)
        self.out_btn.grid(row=0, column=3, padx=4)
        self.export_btn = ttk.Button(bar, text="Export report…", command=self._on_export)
        self.export_btn.grid(row=0, column=4, padx=(4, 14))
        self.scan_label.bind("<Configure>", lambda e: self._refresh_scan_label())

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Full pipeline", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        ttk.Radiobutton(card, text="Classical layers (M2 + M3)", variable=self.engine_var, value="classical", style="Card.TRadiobutton", command=self._on_engine_changed).grid(row=1, column=0, sticky="w", padx=10)
        ttk.Radiobutton(card, text="Spiral fitting (experimental)", variable=self.engine_var, value="spiral", style="Card.TRadiobutton", command=self._on_engine_changed).grid(row=2, column=0, sticky="w", padx=10)
        ttk.Checkbutton(card, text="Scan is already corrected (skip M1)", variable=self.skip_m1, style="Card.TCheckbutton").grid(row=3, column=0, sticky="w", padx=10, pady=(4, 0))
        self.run_all_btn = ttk.Button(card, text="Run full pipeline", style="Accent.TButton", command=self._on_run_all)
        self.run_all_btn.grid(row=4, column=0, sticky="ew", padx=10, pady=(8, 10))

        ttk.Label(parent, text="MODULES — each runs on its own input", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 4))

        m1 = ModuleCard(parent, "M1 · Ingestion", "Removes beam-hardening and ring artifacts, chunk by chunk.", self._on_run_m1)
        m1.add_input("Raw scan (HDF5)", self.v_scan, lambda: self._browse(self.v_scan, "Raw scan"))
        self._cards["m1"] = m1
        m1.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        # M2 + M3 and spiral fitting share one slot; the engine choice swaps them.
        self.classical_group = ttk.Frame(parent)
        self.classical_group.columnconfigure(0, weight=1)
        m2 = ModuleCard(self.classical_group, "M2 · Segmentation", "Splits the volume into sheet segments, breaking apart fused wraps.", self._on_run_m2)
        m2.add_input("Corrected volume", self.v_corrected, lambda: self._browse(self.v_corrected, "Corrected volume"))
        m3 = ModuleCard(self.classical_group, "M3 · Flattening", "Unrolls each segment into a flat page with a coordinate map.", self._on_run_m3)
        m3.add_input("Corrected volume", self.v_corrected, lambda: self._browse(self.v_corrected, "Corrected volume"))
        m3.add_input("Segment labels", self.v_labels, lambda: self._browse(self.v_labels, "Segment labels"))
        self.min_voxels = tk.StringVar(value="auto")
        m3.add_option("Skip segments under (voxels)", lambda p: ttk.Combobox(p, textvariable=self.min_voxels, values=["auto", "0", "100", "1000", "5000"], width=9))
        self.method_var = tk.StringVar(value="auto")
        m3.add_option("Surface model", lambda p: ttk.Combobox(p, textvariable=self.method_var, values=["auto", "cylinder", "patch"], state="readonly", width=9))
        for i, (k, c) in enumerate((("m2", m2), ("m3", m3))):
            self._cards[k] = c
            c.grid(row=i, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        self.spiral_group = ttk.Frame(parent)
        self.spiral_group.columnconfigure(0, weight=1)
        sp = ModuleCard(self.spiral_group, "Spiral fitting", "Fits one global spiral instead of M2 + M3. Experimental; works best on regular wraps.", self._on_run_spiral)
        sp.add_input("Corrected volume", self.v_corrected, lambda: self._browse(self.v_corrected, "Corrected volume"))
        self.omega_var = tk.StringVar()
        sp.add_option("Initial omega (blank = auto)", lambda p: ttk.Entry(p, textvariable=self.omega_var, width=9))
        self.windings_var = tk.StringVar(value="4")
        sp.add_option("Windings to flatten", lambda p: ttk.Spinbox(p, from_=1, to=64, textvariable=self.windings_var, width=9))
        self._cards["spiral_fitting"] = sp
        sp.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        m4 = ModuleCard(parent, "M4 · Ink detection", "Scores every page pixel for ink. Without ink labels this is a density heuristic, not a trained detector.", self._on_run_m4)
        m4.add_input("Flattened pages", self.v_flat, lambda: self._browse(self.v_flat, "Flattened pages"))
        m4.add_input("Scan with ink labels (optional)", self.v_m4_scan, lambda: self._browse(self.v_m4_scan, "Scan"))
        self.v_model = tk.StringVar()
        m4.add_input("Trained ink model (optional)", self.v_model, lambda: self._browse_model())
        self.voxel_var = tk.StringVar()
        m4.add_option("Scan voxel size, µm (blank = unknown)", lambda p: ttk.Entry(p, textvariable=self.voxel_var, width=9))
        self.m4_epochs = tk.StringVar(value="15")
        m4.add_option("Training epochs", lambda p: ttk.Spinbox(p, from_=1, to=500, textvariable=self.m4_epochs, width=9))
        self._cards["m4"] = m4
        m4.grid(row=4, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        m5 = ModuleCard(parent, "M5 · OCR & translation", "Confidence-scored ink regions plus the OCR benchmark. Reads no text from an unseen script.", self._on_run_m5)
        m5.add_input("Ink maps (optional)", self.v_ink, lambda: self._browse(self.v_ink, "Ink maps"))
        self.m5_epochs = tk.StringVar(value="100")
        m5.add_option("Training epochs", lambda p: ttk.Spinbox(p, from_=1, to=1000, textvariable=self.m5_epochs, width=9))
        self._cards["m5"] = m5
        m5.grid(row=5, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))

        self._on_engine_changed()

    def _build_main(self, main: ttk.Frame) -> None:
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.viewer_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.review_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.log_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.viewer_tab, text="Viewer")
        self.notebook.add(self.review_tab, text="Review")
        self.notebook.add(self.log_tab, text="Log")
        self._build_viewer(self.viewer_tab)
        self._build_review(self.review_tab)
        self._build_log(self.log_tab)

    def _build_viewer(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)

        bar = ttk.Frame(tab, style="Card.TFrame")
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        bar.columnconfigure(3, weight=1)
        ttk.Label(bar, text="View", style="Card.TLabel").grid(row=0, column=0, padx=(0, 6))
        self.view_box = ttk.Combobox(bar, textvariable=self.view_var, values=list(VIEWS), state="readonly", width=16)
        self.view_box.grid(row=0, column=1)
        self.view_box.bind("<<ComboboxSelected>>", lambda e: self._request_view())
        ttk.Checkbutton(bar, text="Show boxes", variable=self.show_boxes, style="Card.TCheckbutton", command=self._request_view).grid(row=0, column=2, padx=12)
        ttk.Button(bar, text="Save view as PNG…", style="Small.TButton", command=self._on_save_png).grid(row=0, column=4, sticky="e")

        nav = ttk.Frame(tab, style="Card.TFrame")
        nav.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 0))
        nav.columnconfigure(1, weight=1)
        self.slice_label = ttk.Label(nav, text="Slice 0", style="Card.TLabel", width=16)
        self.slice_scale = ttk.Scale(nav, from_=0, to=1, orient="horizontal", command=self._on_slice_moved)
        self.layer_prev = ttk.Button(nav, text="◀", width=3, style="Small.TButton", command=lambda: self._step_layer(-1))
        self.layer_box = ttk.Combobox(nav, textvariable=self.layer_var, state="readonly", width=18)
        self.layer_next = ttk.Button(nav, text="▶", width=3, style="Small.TButton", command=lambda: self._step_layer(1))
        self.layer_box.bind("<<ComboboxSelected>>", lambda e: self._request_view())
        self._nav = nav
        self._show_nav("none")

        self.notice = WrapLabel(tab, text="", style="Card.TLabel", foreground=WARN, background=WARN_BG, padding=(10, 6), pad=24)
        self.notice.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 0))
        self.notice.grid_remove()

        self.canvas = tk.Canvas(tab, bg="#eef1f5", highlightthickness=1, highlightbackground=BORDER, cursor="crosshair")
        self.canvas.grid(row=3, column=0, sticky="nsew", padx=10, pady=8)
        self.canvas.bind("<Configure>", lambda e: self._schedule_redraw())
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", lambda e: self.info_var.set(self._default_info()))
        self.canvas.bind("<Button-1>", self._on_click)
        self.bind("<Left>", lambda e: self._nudge(-1))
        self.bind("<Right>", lambda e: self._nudge(1))

        self.info_var = tk.StringVar(value="")
        WrapLabel(tab, textvariable=self.info_var, style="CardMuted.TLabel").grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))
        self._draw_empty("Open a scan to see it here.")

    def _build_review(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)
        tab.rowconfigure(5, weight=1)
        self.review_banner = WrapLabel(tab, text="Run M5 to see the review.", style="Card.TLabel", foreground=MUTED, padding=(10, 8), pad=24)
        self.review_banner.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.review_metrics = WrapLabel(tab, text="", style="Card.TLabel", padding=(10, 2), pad=24)
        self.review_metrics.grid(row=1, column=0, sticky="ew", padx=10)
        ttk.Label(tab, text="Pages with ink regions — click a row to open it in the viewer", style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", padx=12, pady=(8, 2))
        tree_frame = ttk.Frame(tab, style="Card.TFrame")
        tree_frame.grid(row=3, column=0, sticky="nsew", padx=10)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=("tokens", "flagged", "coverage"), show="tree headings", height=6)
        self.tree.heading("#0", text="Page")
        self.tree.heading("tokens", text="Regions")
        self.tree.heading("flagged", text="Low confidence")
        self.tree.heading("coverage", text="Ink coverage")
        self.tree.column("#0", width=120, minwidth=80, stretch=True)
        for col in ("tokens", "flagged", "coverage"):
            self.tree.column(col, width=100, minwidth=70, anchor="e", stretch=False)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ttk.Label(tab, text="OCR benchmark on synthetic text lines (not the scan)", style="CardMuted.TLabel").grid(row=4, column=0, sticky="w", padx=12, pady=(8, 2))
        text_frame = ttk.Frame(tab, style="Card.TFrame")
        text_frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.review_text = tk.Text(text_frame, height=6, wrap="word", state="disabled", bg=SURFACE, fg=TEXT, relief="flat", highlightthickness=1, highlightbackground=BORDER, font=(FONT, 10))
        self.review_text.tag_configure("muted", foreground=MUTED)
        text_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.review_text.yview)
        self.review_text.configure(yscrollcommand=text_scroll.set)
        self.review_text.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")

    def _build_log(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.log_text = tk.Text(tab, wrap="word", state="disabled", bg=SURFACE, fg=TEXT, relief="flat", highlightthickness=0, font=("Consolas", 9))
        self.log_text.tag_configure("err", foreground=ERR)
        self.log_text.tag_configure("muted", foreground=MUTED)
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scroll.grid(row=0, column=1, sticky="ns", pady=10)

    # ------------------------------------------------------------------ small helpers

    def _on_engine_changed(self) -> None:
        if self.engine_var.get() == "classical":
            self.spiral_group.grid_remove()
            self.classical_group.grid(row=3, column=0, sticky="ew")
        else:
            self.classical_group.grid_remove()
            self.spiral_group.grid(row=3, column=0, sticky="ew")

    def _show_nav(self, mode: str) -> None:
        for w in (self.slice_label, self.slice_scale, self.layer_prev, self.layer_box, self.layer_next):
            w.grid_forget()
        if mode == "slice":
            self.slice_label.grid(row=0, column=0, padx=(0, 8))
            self.slice_scale.grid(row=0, column=1, sticky="ew")
        elif mode == "layer":
            self.layer_prev.grid(row=0, column=0)
            self.layer_box.grid(row=0, column=1, sticky="w", padx=4)
            self.layer_next.grid(row=0, column=2)

    def _refresh_scan_label(self) -> None:
        text = self.v_scan.get()
        if not text:
            self.scan_label.configure(text="No scan loaded")
            return
        width = max(40, self.scan_label.winfo_width() - 8)
        font = tkfont.Font(font=(FONT, 9))
        shown = text
        while font.measure(shown) > width and len(shown) > 6:
            shown = vm.middle_truncate(text, len(shown) - 3)
        self.scan_label.configure(text=shown)

    def _set_auto(self, var: tk.StringVar, value) -> None:
        """Fill a downstream input unless the user typed their own path into it."""
        if value is None:
            return
        current = var.get()
        if not current or self._auto_values.get(str(var)) == current:
            var.set(str(value))
            self._auto_values[str(var)] = str(value)

    def _browse(self, var: tk.StringVar, title: str) -> None:
        start = Path(var.get()).parent if var.get() else None
        path = filedialog.askopenfilename(title=title, filetypes=H5_TYPES, initialdir=str(start) if start and start.exists() else None)
        if not path:
            return
        var.set(path)
        self._auto_values.pop(str(var), None)
        if var is self.v_scan:
            self._after_scan_set(Path(path))
        else:
            self._request_view(reset_position=True)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(title="Trained ink model", filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self.v_model.set(path)

    def _log(self, phase: str, message: str) -> None:
        self._ui_queue.put(("log", phase, message))

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n", tag or ())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool, label: str = "") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for w in [self.open_btn, self.out_btn, self.run_all_btn] + [c.button for c in self._cards.values()]:
            w.configure(state=state)
        if busy:
            self.status_var.set(label)
            self.progress.grid()
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.grid_remove()

    # ------------------------------------------------------------------ scan handling

    def _on_open_scan(self) -> None:
        path = filedialog.askopenfilename(title="Open scan (HDF5)", filetypes=H5_TYPES)
        if path:
            self.v_scan.set(path)
            self._auto_values.pop(str(self.v_scan), None)
            self._after_scan_set(Path(path))

    def _after_scan_set(self, path: Path) -> None:
        self.orchestrator = None
        if not self.v_out.get() or self._auto_values.get(str(self.v_out)) == self.v_out.get():
            self.v_out.set(str(path.parent / "gui_run"))
            self._auto_values[str(self.v_out)] = self.v_out.get()
        self._set_auto(self.v_m4_scan, path)
        self._refresh_scan_label()
        self._append_log(f"[Load] {path}", "muted")
        self.status_var.set(f"Loaded {path.name} — results go to {self.v_out.get()}")
        self.view_var.set("Scan slice")
        self._request_view(reset_position=True)

    def _on_choose_output(self) -> None:
        path = filedialog.askdirectory(title="Output folder", initialdir=self.v_out.get() or None)
        if path:
            self.v_out.set(path)
            self._auto_values.pop(str(self.v_out), None)
            if self.orchestrator:
                self.orchestrator.set_work_dir(path)
            self.status_var.set(f"Results will be written to {path}")

    def _ensure_orchestrator(self) -> PipelineOrchestrator:
        out = self.v_out.get() or str(Path.cwd() / "gui_run")
        if self.orchestrator is None:
            self.orchestrator = PipelineOrchestrator(out)
        elif Path(out) != self.orchestrator.work_dir:
            self.orchestrator.set_work_dir(out)
        return self.orchestrator

    # ------------------------------------------------------------------ running modules

    def _need(self, key: str, *required: tuple[str, str]) -> bool:
        card = self._cards[key]
        for label, value in required:
            if not value.strip():
                card.show_error(f"Choose a file for “{label}”, or run the earlier module first.")
                return False
            if not Path(value).exists():
                card.show_error(f"File not found: {value}")
                return False
        card.show_error(None)
        return True

    def _int_option(self, key: str, var: tk.StringVar, label: str, minimum: int = 0) -> int | None:
        try:
            value = int(var.get())
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            self._cards[key].show_error(f"“{label}” must be a whole number of at least {minimum}.")
            return None

    def _begin(self, key: str, label: str, fn) -> None:
        """Run ``fn(orchestrator)`` on a worker thread; results come back through the UI queue."""
        if self._busy:
            return
        orch = self._ensure_orchestrator()
        if key in self._cards:
            self._cards[key].set_status("running")
            self._cards[key].show_error(None)
        self._set_busy(True, label)
        self.notebook.select(self.viewer_tab)

        def target():
            try:
                self._ui_queue.put(("phase_done", key, fn(orch)))
            except Exception as exc:  # noqa: BLE001 - UI error boundary
                self._ui_queue.put(("phase_crash", key, str(exc)))

        threading.Thread(target=target, daemon=True).start()

    def _on_run_m1(self) -> None:
        if not self._need("m1", ("Raw scan", self.v_scan.get())):
            return
        scan = self.v_scan.get()
        self._begin("m1", "Running M1 — correcting artifacts…", lambda o: o.run_ingestion(scan, self._log))

    def _on_run_m2(self) -> None:
        if not self._need("m2", ("Corrected volume", self.v_corrected.get())):
            return
        corrected = self.v_corrected.get()
        self._begin("m2", "Running M2 — segmenting layers…", lambda o: o.run_segmentation(None, self._log, input_path=corrected))

    def _on_run_m3(self) -> None:
        if not self._need("m3", ("Corrected volume", self.v_corrected.get()), ("Segment labels", self.v_labels.get())):
            return
        text = self.min_voxels.get().strip().lower()
        min_voxels = None if text in ("", "auto") else self._int_option("m3", self.min_voxels, "Skip segments under (voxels) — a number or auto")
        if min_voxels is None and text not in ("", "auto"):
            return
        corrected, labels, method = self.v_corrected.get(), self.v_labels.get(), self.method_var.get()
        self._begin("m3", "Running M3 — flattening segments…", lambda o: o.run_flattening(self._log, corrected, labels, min_voxels, method))

    def _spiral_options(self) -> tuple[float | None, int] | None:
        text = self.omega_var.get().strip()
        try:
            omega = float(text) if text else None
            if omega is not None and omega <= 0:
                raise ValueError
        except ValueError:
            self._cards["spiral_fitting"].show_error("Initial omega must be a positive number, or blank to estimate it.")
            return None
        windings = self._int_option("spiral_fitting", self.windings_var, "Windings to flatten", 1)
        return None if windings is None else (omega, windings)

    def _on_run_spiral(self) -> None:
        if not self._need("spiral_fitting", ("Corrected volume", self.v_corrected.get())):
            return
        opts = self._spiral_options()
        if opts is None:
            return
        omega, windings = opts
        corrected = self.v_corrected.get()
        self._begin("spiral_fitting", "Running spiral fitting…", lambda o: o.run_spiral_fitting(omega, windings, callback=self._log, input_path=corrected))

    def _on_run_m4(self) -> None:
        if not self._need("m4", ("Flattened pages", self.v_flat.get())):
            return
        scan = self.v_m4_scan.get().strip() or self.v_scan.get().strip() or self.v_corrected.get().strip()
        if not scan or not Path(scan).exists():
            self._cards["m4"].show_error("M4 needs a scan to read pixel values from: choose one under “Scan with ink labels”.")
            return
        epochs = self._int_option("m4", self.m4_epochs, "Training epochs", 1)
        if epochs is None:
            return
        model = self.v_model.get().strip() or None
        if model and not Path(model).exists():
            self._cards["m4"].show_error(f"Model file not found: {model}")
            return
        voxel = None
        if self.voxel_var.get().strip():
            try:
                voxel = float(self.voxel_var.get())
                if voxel <= 0:
                    raise ValueError
            except ValueError:
                self._cards["m4"].show_error("Scan voxel size must be a positive number of micrometres, or blank.")
                return
        flat = self.v_flat.get()
        self._begin(
            "m4", "Running M4 — detecting ink…",
            lambda o: o.run_ink_detection(scan, self._log, flattened_path=flat, epochs=epochs, model_path=model, voxel_size_um=voxel),
        )

    def _on_run_m5(self) -> None:
        ink = self.v_ink.get().strip()
        if ink and not self._need("m5", ("Ink maps", ink)):
            return
        epochs = self._int_option("m5", self.m5_epochs, "Training epochs", 1)
        if epochs is None:
            return
        self._cards["m5"].show_error(None)
        self._begin("m5", "Running M5 — OCR benchmark and ink regions…", lambda o: o.run_ocr_translation(self._log, ink_path=ink or None, epochs=epochs))

    def _on_run_all(self) -> None:
        if not self._need("m1", ("Raw scan", self.v_scan.get())):
            return
        engine, scan, skip = self.engine_var.get(), self.v_scan.get(), self.skip_m1.get()
        omega, windings = None, 4
        if engine == "spiral":
            opts = self._spiral_options()
            if opts is None:
                return
            omega, windings = opts
        self._begin(
            "all",
            "Running the full pipeline…",
            lambda o: o.run_all(scan, self._log, engine=engine, spiral_initial_omega=omega, spiral_n_windings=windings, skip_ingestion=skip),
        )

    # ------------------------------------------------------------------ results

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, *payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(f"[{payload[0]}] {payload[1]}")
                elif kind == "view":
                    token, result = payload
                    if token == self._request_token:
                        self._apply_view(result)
                elif kind == "phase_done":
                    self._on_phase_done(*payload)
                elif kind == "phase_crash":
                    self._finish(payload[0], False, payload[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _on_phase_done(self, key: str, result) -> None:
        results = result if key == "all" else [result]
        for r in results:
            self._record_result(r)
        failed = next((r for r in results if not r.ok), None)
        self._finish(key, failed is None, failed.message if failed else "Done.")
        if failed is None:
            self._after_results(key)

    def _record_result(self, result) -> None:
        orch = self.orchestrator
        key = next((k for k, r in orch.results.items() if r is result), None) if orch else None
        card = self._cards.get(key or "")
        if card:
            card.set_status("done" if result.ok else "failed")
            card.show_error(None if result.ok else result.message)
        self._append_log(f"[{result.name}] {'OK' if result.ok else 'FAILED'} — {result.message}", None if result.ok else "err")
        if not result.ok and result.details.get("traceback"):
            self._append_log(result.details["traceback"], "muted")

    def _finish(self, key: str, ok: bool, message: str) -> None:
        self._set_busy(False)
        card = self._cards.get(key)
        if card:
            card.set_status("done" if ok else "failed")
            if not ok:
                card.show_error(message)
        self.status_var.set(message if ok else f"Stopped: {message}")
        if not ok:
            self._append_log(f"Stopped: {message}", "err")
            self.notebook.select(self.log_tab)

    def _after_results(self, key: str) -> None:
        orch = self.orchestrator
        res = orch.results
        if "m1" in res and res["m1"].ok:
            self._set_auto(self.v_corrected, res["m1"].output_path)
            self._set_auto(self.v_m4_scan, res["m1"].output_path)  # pages were cut from the corrected volume
        if "m2" in res and res["m2"].ok:
            self._set_auto(self.v_labels, res["m2"].output_path)
        flat = orch.flattened_result()
        if flat:
            self._set_auto(self.v_flat, flat.output_path)
        if "m4" in res and res["m4"].ok:
            self._set_auto(self.v_ink, res["m4"].output_path)

        notices = []
        m2 = res.get("m2")
        if m2 and m2.ok and m2.details.get("fused_warning"):
            notices.append("M2: one segment still holds most of the scan, so layers are probably still fused.")
        sp = res.get("spiral_fitting")
        if sp and sp.ok:
            notices += [f"Spiral fit: {w}." for w in sp.details.get("warnings", [])]
        self._pipeline_notice = " ".join(notices) or None

        m5 = res.get("m5")
        if key in ("m5", "all") and m5 and m5.ok and m5.output_path and Path(m5.output_path).exists():
            self._render_review(json.loads(Path(m5.output_path).read_text()))
            if key == "m5":
                self.notebook.select(self.review_tab)
                return
        target = PHASE_VIEW.get(key)
        if key == "all":
            target = "Ink detection" if "m4" in res and res["m4"].ok else ("Flattened page" if flat else "Segmentation")
        if target:
            self.view_var.set(target)
            self.notebook.select(self.viewer_tab)
            self._request_view(reset_position=True)

    # ------------------------------------------------------------------ viewer

    def _viewer_paths(self) -> dict:
        def p(var):
            text = var.get().strip()
            return Path(text) if text and Path(text).exists() else None

        return {"scan": p(self.v_scan), "corrected": p(self.v_corrected), "labels": p(self.v_labels), "flattened": p(self.v_flat), "ink": p(self.v_ink)}

    def _request_view(self, reset_position: bool = False) -> None:
        """Build the current view on a worker thread (slices of a multi-GB scan take a moment)."""
        kind = VIEWS[self.view_var.get()]
        paths = self._viewer_paths()
        z = int(self.z_var.get())
        layer = None if reset_position else (self.layer_var.get() or None)
        show_extent = self.show_boxes.get()
        self._request_token += 1
        token = self._request_token

        def work():
            try:
                zz = z
                depth_src = paths.get("corrected") if kind == "segmentation" else (paths.get("scan") or paths.get("corrected"))
                if reset_position and depth_src and kind in ("scan", "segmentation"):
                    zz = vm.volume_depth(depth_src) // 2  # the middle slice is the most representative
                self._ui_queue.put(("view", token, (vm.build_view(kind, paths, zz, layer, show_extent), zz)))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(("view", token, exc))

        threading.Thread(target=work, daemon=True).start()

    def _apply_view(self, result) -> None:
        if isinstance(result, Exception):
            self._view = None
            self._show_nav("none")
            self.notice.grid_remove()
            self._draw_empty(str(result) if isinstance(result, ValueError) else f"Couldn't show this view: {result}")
            return
        view, z = result
        self._view, self._selected = view, None
        self._base_images.clear()
        if view.depth:
            self._show_nav("slice")
            self.z_var.set(z)
            self.slice_scale.configure(to=max(1, view.depth - 1))
            self.slice_scale.set(z)
            self.slice_label.configure(text=f"Slice {z} / {view.depth - 1}")
        elif view.layers:
            self._show_nav("layer")
            self.layer_box.configure(values=view.layers)
            if self.layer_var.get() not in view.layers:
                self.layer_var.set(view.layers[0])
        else:
            self._show_nav("none")
        notice = view.notice or self._pipeline_notice
        if notice:
            self.notice.configure(text=notice)
            self.notice.grid()
        else:
            self.notice.grid_remove()
        self.info_var.set(self._default_info())
        self._redraw()

    def _default_info(self) -> str:
        return "" if self._view is None else self._view.caption + "   ·   hover for values, click a box for details"

    def _on_slice_moved(self, value) -> None:
        z = int(float(value))
        depth = self._view.depth if self._view and self._view.depth else None
        self.slice_label.configure(text=f"Slice {z}" + (f" / {depth - 1}" if depth else ""))
        if z == self.z_var.get():
            return
        self.z_var.set(z)
        if self._load_job:
            self.after_cancel(self._load_job)
        self._load_job = self.after(80, self._request_view)

    def _nudge(self, delta: int) -> None:
        if isinstance(self.focus_get(), (ttk.Entry, ttk.Spinbox, ttk.Combobox, tk.Text)):
            return
        if self._view and self._view.depth:
            self.slice_scale.set(min(max(0, self.z_var.get() + delta), self._view.depth - 1))
        elif self._view and self._view.layers:
            self._step_layer(delta)

    def _step_layer(self, delta: int) -> None:
        layers = self._view.layers if self._view else None
        if not layers or self.layer_var.get() not in layers:
            return
        self.layer_var.set(layers[min(max(0, layers.index(self.layer_var.get()) + delta), len(layers) - 1)])
        self._request_view()

    def _schedule_redraw(self) -> None:
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(30, self._redraw)

    def _draw_empty(self, message: str) -> None:
        self.canvas.delete("all")
        self._placed = []
        w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        self.canvas.create_text(w / 2, h / 2, text=message, fill=MUTED, font=(FONT, 11), width=max(120, w - 60), justify="center")
        self.info_var.set("")

    def _redraw(self) -> None:
        self._redraw_job = None
        if self._view is None:
            self._draw_empty("Open a scan to see it here.")
            return
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self._placed = vm.fit_layout([p.shape for p in self._view.panels], w, h)
        self._photos.clear()
        boxes_on = self.show_boxes.get()
        for placed in self._placed:
            panel = self._view.panels[placed.index]
            base = self._base_images.get(placed.index)
            if base is None:
                base = self._base_images[placed.index] = vm.panel_image(panel)
            size = (max(1, int(round(placed.w))), max(1, int(round(placed.h))))
            photo = ImageTk.PhotoImage(base.resize(size, Image.LANCZOS if placed.scale < 1 else Image.BILINEAR))
            self._photos.append(photo)
            x, y = placed.x, placed.y
            self.canvas.create_text(x, y - 6, text=vm.middle_truncate(panel.title, max(10, int(placed.w / 7))), anchor="sw", fill=TEXT, font=(FONT, 10, "bold"))
            self.canvas.create_image(x, y, anchor="nw", image=photo)
            self.canvas.create_rectangle(x, y, x + placed.w, y + placed.h, outline=BORDER)
            if boxes_on:
                for box in panel.boxes:
                    self._draw_box(placed, box, self._selected is not None and self._selected[1] is box)

    def _draw_box(self, placed: vm.Placed, box: vm.Box, selected: bool) -> None:
        s = placed.scale
        x0, y0 = placed.x + box.x0 * s, placed.y + box.y0 * s
        x1, y1 = placed.x + box.x1 * s, placed.y + box.y1 * s
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=box.color, width=3 if selected else 2)
        if box.label and (x1 - x0) >= 30 and (y1 - y0) >= 16:
            tag = self.canvas.create_text(x0 + 4, y0 + 3, text=box.label, anchor="nw", fill="#ffffff", font=(FONT, 8, "bold"))
            bb = self.canvas.bbox(tag)
            if bb:
                bg = self.canvas.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1, fill=box.color, outline="")
                self.canvas.tag_lower(bg, tag)

    def _on_hover(self, event) -> None:
        if self._view is None:
            return
        hit = vm.hit_test(self._placed, event.x, event.y)
        if hit is None:
            self.info_var.set(self._default_info())
            return
        idx, ix, iy = hit
        panel = self._view.panels[idx]
        box = vm.box_at(panel, ix, iy) if self.show_boxes.get() else None
        self.info_var.set(vm.readout(panel, ix, iy) + (f"   ·   {box.info}" if box and box.info else ""))

    def _on_click(self, event) -> None:
        if self._view is None:
            return
        hit = vm.hit_test(self._placed, event.x, event.y)
        self._selected = None
        if hit:
            idx, ix, iy = hit
            box = vm.box_at(self._view.panels[idx], ix, iy) if self.show_boxes.get() else None
            if box:
                self._selected = (idx, box)
                self.info_var.set(box.info or box.label)
        self._redraw()

    def _on_save_png(self) -> None:
        if self._view is None:
            messagebox.showinfo("Nothing to save", "There is no view to save yet.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if dest:
            size = (max(400, self.canvas.winfo_width()), max(300, self.canvas.winfo_height()))
            vm.compose_png(self._view, size, show_boxes=self.show_boxes.get()).save(dest)
            self.status_var.set(f"Saved {dest}")

    # ------------------------------------------------------------------ review

    def _render_review(self, report: dict) -> None:
        scan = report.get("scan_transcription", {})
        if scan and not scan.get("available", True):
            text = "No text was read from this scan. " + scan.get("reason", "")
            if scan.get("warning"):
                text += "  " + scan["warning"]
            self.review_banner.configure(text=text, foreground=WARN, background=WARN_BG)
        else:
            self.review_banner.configure(text="Transcription available.", foreground=OK, background=SURFACE)
        parts = [f"Synthetic benchmark — CER {report.get('mean_cer', float('nan')):.3f}, WER {report.get('mean_wer', float('nan')):.3f}"]
        if "total_tokens" in scan:
            parts.append(
                f"{scan['total_tokens']:,} ink regions on {scan.get('pages_with_ink_tokens', 0):,} of {scan.get('n_pages', 0):,} pages, "
                f"{scan.get('total_flagged_low_confidence', 0):,} low-confidence"
            )
        self.review_metrics.configure(text="   |   ".join(parts))

        self.tree.delete(*self.tree.get_children())
        pages = report.get("ink_map_tokens", {})
        for layer in sorted(pages, key=vm.natural_key):
            info = pages[layer]
            if info["n_tokens"]:
                self.tree.insert("", "end", iid=layer, text=layer, values=(info["n_tokens"], info["n_flagged"], f"{info.get('ink_coverage', 0):.1%}"))

        self.review_text.configure(state="normal")
        self.review_text.delete("1.0", "end")
        for sample in report.get("samples", []):
            self.review_text.insert("end", f"target     {sample['target']}\ncorrected  {sample['corrected']}\ntranslated {sample['translation']}\n")
            self.review_text.insert("end", f"{sample['n_flagged']} low-confidence character(s) infilled\n\n", "muted")
        self.review_text.configure(state="disabled")

    def _on_tree_select(self, _event) -> None:
        sel = self.tree.selection()
        if sel:
            self.layer_var.set(sel[0])
            self.view_var.set("Ink detection")
            self.notebook.select(self.viewer_tab)
            self._request_view()

    # ------------------------------------------------------------------ export

    def _on_export(self) -> None:
        if self.orchestrator is None or not self.orchestrator.results:
            messagebox.showinfo("Nothing to export", "Run at least one module first.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not dest:
            return
        summary = {
            phase: {"ok": r.ok, "output_path": str(r.output_path) if r.output_path else None, "message": r.message}
            for phase, r in self.orchestrator.results.items()
        }
        Path(dest).write_text(json.dumps(summary, indent=2))
        self.status_var.set(f"Exported {dest}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
