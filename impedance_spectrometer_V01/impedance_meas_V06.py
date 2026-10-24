#!/usr/bin/env python3
"""
Impedance Spectrograph V0.6

This version includes:
- sparse multi-tone transmit signal with a 21.096 kHz top tone
- FFT-size control that only applies after Enter or button press
- calibration workflow and ratio-only mode
- level bargraphs using actual float audio amplitudes
- menu bar with File / Soundcard / About
- JSON save/load of configuration and calibration
- soundcard selection limited to devices with >=2 input and >=2 output channels
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QDoubleValidator, QFont, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


APP_NAME = "Impedance spectrograph"
APP_VERSION = "V0.6"
APP_TITLE = f"{APP_NAME} {APP_VERSION}"

SAMPLE_RATE = 48000
FFT_SIZE = 48000
BUFFER_SEND_LEN = 2 * FFT_SIZE
BUFFER_RECV_LEN = 3 * FFT_SIZE

DAC_SPAN_PEAK = 32767.0
TEST_SIGNAL_AMPLITUDE = (120.0 / 128.0) * DAC_SPAN_PEAK
ADC_FULL_SCALE_PP = 2.0

FREQ_RATIO_MAX = 1.0594631
FREQ_START_HZ = 21096.0
REF_RESISTANCE = 100.0


# -----------------------------------------------------------------------------
# Helpers for FFT-size acceptance and normalization
# -----------------------------------------------------------------------------

def prime_factors(n: int) -> list[int]:
    """Return the prime factor list of n."""
    factors: list[int] = []
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            factors.append(p)
            x //= p
        p += 1 if p == 2 else 2
    if x > 1:
        factors.append(x)
    return factors


def greatest_prime_factor(n: int) -> int:
    """Return the largest prime factor, or 1 for n <= 1."""
    pf = prime_factors(n)
    return max(pf) if pf else 1


def next_power_of_two_up(x: int) -> int:
    """Return the next power of two >= x."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def next_power_of_two_down(x: int) -> int:
    """Return the largest power of two <= x."""
    if x <= 1:
        return 1
    return 1 << ((x).bit_length() - 1)


def normalize_fft_size(candidate: int) -> int:
    """
    Normalize FFT size to a valid value in [256, 262144].
    Any candidate whose largest prime factor exceeds 7 is rounded up to the next
    valid power-of-two.
    """
    lo, hi = 256, 262144
    if candidate < lo:
        candidate = lo
    if candidate > hi:
        candidate = hi

    # Keep the value at the closest allowed power-of-two if prime factor constraint fails.
    if candidate & (candidate - 1):
        candidate = next_power_of_two_up(candidate)

    while candidate > hi:
        candidate //= 2

    while candidate < lo:
        candidate *= 2

    while greatest_prime_factor(candidate) > 7:
        candidate = next_power_of_two_up(candidate + 1)
        if candidate > hi:
            return hi

    return int(candidate)


class AllowedFFTSizeSpinBox(QSpinBox):
    """Spin box that only steps through valid FFT sizes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(256, 262144)
        self.setSingleStep(1)
        self.setValue(FFT_SIZE)

    def stepBy(self, steps: int):
        current = self.value()
        if steps > 0:
            nxt = current
            while True:
                nxt += 1
                if nxt > 262144:
                    nxt = 262144
                    break
                if normalize_fft_size(nxt) > current:
                    break
            self.setValue(normalize_fft_size(nxt))
        elif steps < 0:
            nxt = current
            while True:
                nxt -= 1
                if nxt < 256:
                    nxt = 256
                    break
                if normalize_fft_size(nxt) < current:
                    break
            self.setValue(normalize_fft_size(nxt))


# -----------------------------------------------------------------------------
# Complex calibration / helper functions
# -----------------------------------------------------------------------------

@dataclass
class CalibrationCoefficients:
    A: np.ndarray | complex = None
    B: np.ndarray | complex = None
    C: np.ndarray | complex = None
    D: np.ndarray | complex = None

    def __post_init__(self):
        if self.A is None:
            self.A = np.ones(1, dtype=np.complex128)
        if self.B is None:
            self.B = np.zeros(1, dtype=np.complex128)
        if self.C is None:
            self.C = np.zeros(1, dtype=np.complex128)
        if self.D is None:
            self.D = np.ones(1, dtype=np.complex128)


def compute_sparse_frequencies(sample_rate: int = SAMPLE_RATE,
                              fft_size: int = FFT_SIZE,
                              start_hz: float = FREQ_START_HZ,
                              max_ratio: float = FREQ_RATIO_MAX) -> list[int]:
    """Compute sparse FFT bins with a maximum tone ratio and a 50 Hz skip."""
    bin_size = sample_rate / fft_size
    start_bin = int(round(start_hz / bin_size))
    start_bin = max(1, min(start_bin, fft_size // 2 - 1))

    bins: list[int] = []
    current_bin = start_bin
    while current_bin >= 1:
        bins.append(current_bin)
        next_bin = int(np.ceil(current_bin / max_ratio))
        if next_bin == current_bin or next_bin < 10:
            break
        if next_bin == 50:
            if current_bin == 51:
                next_bin = 49
            else:
                next_bin = 51
        current_bin = next_bin
    return sorted(set(bins))


def schroeder_phase_sequence(n: int) -> np.ndarray:
    """Generate a Schroeder phase sequence."""
    if n <= 0:
        return np.array([], dtype=np.float64)
    return np.array([-np.pi * k * (k + 1) / n for k in range(n)], dtype=np.float64)


def synthesize_multitone_fft(fft_size: int, bins: list[int]) -> np.ndarray:
    """Synthesize a sparse multi-tone spectrum with conjugate symmetry."""
    if not bins:
        return np.zeros(fft_size, dtype=np.complex128)

    spectrum = np.zeros(fft_size, dtype=np.complex128)
    phases = schroeder_phase_sequence(len(bins))
    amplitude = TEST_SIGNAL_AMPLITUDE / np.sqrt(len(bins))

    for idx, bin_idx in enumerate(bins):
        phase = phases[idx]
        spectrum[bin_idx] = amplitude * np.exp(1j * phase)
        if bin_idx != 0 and bin_idx != fft_size // 2:
            mirror = fft_size - bin_idx
            spectrum[mirror] = amplitude * np.exp(-1j * phase)

    return spectrum


def generate_multitone_time_signal(sample_count: int, bins: list[int]) -> np.ndarray:
    """Generate time-domain test signal from inverse FFT."""
    spectrum = synthesize_multitone_fft(FFT_SIZE, bins)
    time_signal = np.fft.ifft(spectrum)
    time_signal = np.real(time_signal)

    peak = np.max(np.abs(time_signal))
    if peak > 0:
        time_signal = time_signal * (TEST_SIGNAL_AMPLITUDE / peak)

    if sample_count > len(time_signal):
        time_signal = np.pad(time_signal, (0, sample_count - len(time_signal)))
    elif sample_count < len(time_signal):
        time_signal = time_signal[:sample_count]

    return time_signal.astype(np.float32)


def apply_calibration_coefficients(v_open: np.ndarray,
                                  v_short: np.ndarray,
                                  v_load: np.ndarray,
                                  ref_resistance: float) -> CalibrationCoefficients:
    """Calculate complex calibration constants from open/short/load results."""
    v_open = np.asarray(v_open, dtype=np.complex128)
    v_short = np.asarray(v_short, dtype=np.complex128)
    v_load = np.asarray(v_load, dtype=np.complex128)

    open_valid = np.abs(v_open) > 1e-12
    denom = v_load - v_short
    denom_valid = np.abs(denom) > 1e-12

    if not np.all(open_valid):
        raise ValueError("Open calibration contains near-zero values.")
    if not np.all(denom_valid):
        raise ValueError("Load and short calibration are too similar for one or more bins.")

    C = np.empty_like(v_open, dtype=np.complex128)
    A = np.empty_like(v_open, dtype=np.complex128)
    B = np.empty_like(v_open, dtype=np.complex128)
    D = np.ones_like(v_open, dtype=np.complex128)

    C[:] = -1.0 / v_open
    A[:] = ref_resistance * (1.0 - v_load / v_open) / denom
    B[:] = -A * v_short

    return CalibrationCoefficients(A=A, B=B, C=C, D=D)


def default_calibration() -> CalibrationCoefficients:
    return CalibrationCoefficients(
        A=np.ones(1, dtype=np.complex128),
        B=np.zeros(1, dtype=np.complex128),
        C=np.zeros(1, dtype=np.complex128),
        D=np.ones(1, dtype=np.complex128),
    )


def coeffs_are_zero_pair(a: complex, b: complex) -> bool:
    return abs(a) < 1e-12 and abs(b) < 1e-12


def normalize_calibration_coeffs_single_point(A: complex, B: complex, C: complex, D: complex) -> tuple[complex, complex, complex, complex]:
    """Reject invalid frequency calibration entries and return default coefficients."""
    if (coeffs_are_zero_pair(A, C) and coeffs_are_zero_pair(B, D)) or (coeffs_are_zero_pair(A, B) and coeffs_are_zero_pair(C, D)):
        return 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j
    return A, B, C, D


def sanitize_bins(bins: list[int], fft_size: int) -> list[int]:
    """Remove doubles, sort, and check valid indices range."""
    cleaned = []
    for b in bins:
        if not isinstance(b, int):
            continue
        if 1 <= b <= fft_size // 2 - 1:
            cleaned.append(int(b))
    cleaned = sorted(set(cleaned))
    if len(cleaned) < 2:
        return sorted(set([max(1, fft_size // 4), max(1, fft_size // 8)]))
    return cleaned


def get_soundcard_devices() -> list[tuple[int, str]]:
    """Return only devices with at least 2 input and 2 output channels."""
    devices: list[tuple[int, str]] = []
    try:
        count = sd.device_count()
    except Exception:
        return devices

    for idx in range(count):
        try:
            info = sd.query_devices(idx)
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        max_in = int(info.get("max_input_channels", 0) or 0)
        max_out = int(info.get("max_output_channels", 0) or 0)
        if max_in >= 2 and max_out >= 2:
            name = str(info.get("name", f"device_{idx}"))
            devices.append((idx, name))
    return devices


# -----------------------------------------------------------------------------
# Ring buffer and audio stream manager
# -----------------------------------------------------------------------------

class RingBuffer:
    def __init__(self, capacity: int, num_channels: int = 1):
        self.capacity = capacity
        self.num_channels = num_channels
        self.buffer = np.zeros((capacity, num_channels), dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray):
        with self.lock:
            samples = len(data)
            if samples == 0:
                return
            space_to_end = self.capacity - self.write_pos
            if samples <= space_to_end:
                self.buffer[self.write_pos:self.write_pos + samples] = data
            else:
                self.buffer[self.write_pos:] = data[:space_to_end]
                self.buffer[:samples - space_to_end] = data[space_to_end:]
            self.write_pos = (self.write_pos + samples) % self.capacity

    def read(self, start_pos: int, length: int) -> np.ndarray:
        with self.lock:
            if length <= 0:
                return np.empty((0, self.num_channels), dtype=np.float32)
            pos = start_pos % self.capacity
            end = (pos + length) % self.capacity
            if pos < end:
                return self.buffer[pos:end].copy()
            if pos + length <= self.capacity:
                return self.buffer[pos:pos + length].copy()
            return np.vstack((self.buffer[pos:], self.buffer[:end])).copy()

    def get_write_pos(self) -> int:
        with self.lock:
            return self.write_pos


class AudioStreamManager:
    def __init__(self, device_id: int, sample_rate: int, fft_size: int, bins: list[int]):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bins = list(bins)
        self.stream = None

        self.send_buffer = RingBuffer(BUFFER_SEND_LEN, num_channels=1)
        self.recv_left = RingBuffer(BUFFER_RECV_LEN, num_channels=1)
        self.recv_right = RingBuffer(BUFFER_RECV_LEN, num_channels=1)

        test_signal = generate_multitone_time_signal(self.fft_size, self.bins).astype(np.float32)
        test_signal = test_signal / 32767.0
        self.send_buffer.write(test_signal.reshape(-1, 1))
        self.send_buffer.write(test_signal.reshape(-1, 1))

        self.total_samples = 0
        self.recv_write_pos = 0
        self.dsp_read_pos = 0
        self.lock = threading.Lock()

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        send_data = self.send_buffer.read(self.total_samples % BUFFER_SEND_LEN, frames)
        if send_data.shape[0] < frames:
            pad = np.zeros((frames - send_data.shape[0], 1), dtype=np.float32)
            send_data = np.vstack((send_data, pad))

        outdata[:, 0] = send_data[:frames, 0]
        outdata[:, 1] = send_data[:frames, 0]

        with self.lock:
            self.recv_left.write(indata[:, 0].reshape(-1, 1))
            self.recv_right.write(indata[:, 1].reshape(-1, 1))
            self.recv_write_pos = self.recv_left.get_write_pos()
            self.total_samples += frames

    def start(self):
        self.stream = sd.Stream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=2,
            blocksize=4096,
            callback=self.audio_callback,
            latency="low",
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# -----------------------------------------------------------------------------
# DSP worker
# -----------------------------------------------------------------------------

class DSPWorker(QObject):
    spectrum_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, stream_manager: AudioStreamManager, sparse_bins: list[int]):
        super().__init__()
        self.stream_manager = stream_manager
        self.sparse_bins = list(sparse_bins)
        self.sparse_bins_array = np.array(self.sparse_bins, dtype=np.int64)
        self.running = False

        self.fft_size = stream_manager.fft_size
        self.sample_rate = stream_manager.sample_rate
        self.calibration = default_calibration()
        self.last_V = None
        self.last_recv_pos = 0
        self.fft_count = 0

    @staticmethod
    def compute_pp_dbfs(data: np.ndarray) -> float:
        pp = float(np.ptp(data))
        if pp <= 0:
            return -120.0
        return 20.0 * np.log10(pp / ADC_FULL_SCALE_PP)

    def process_fft_block(self):
        with self.stream_manager.lock:
            recv_write_pos = self.stream_manager.recv_write_pos
            available = (recv_write_pos - self.last_recv_pos) % BUFFER_RECV_LEN

        if available < self.fft_size:
            return

        left_data = self.stream_manager.recv_left.read(self.last_recv_pos, self.fft_size)
        right_data = self.stream_manager.recv_right.read(self.last_recv_pos, self.fft_size)
        self.last_recv_pos = (self.last_recv_pos + self.fft_size) % BUFFER_RECV_LEN

        left_spectrum = np.fft.fft(left_data[:, 0])
        right_spectrum = np.fft.fft(right_data[:, 0])

        left_vals = left_spectrum[self.sparse_bins_array]
        right_vals = right_spectrum[self.sparse_bins_array]
        frequencies = self.sparse_bins_array * (self.sample_rate / self.fft_size)

        V = np.empty_like(left_vals, dtype=np.complex128)
        valid = np.abs(right_vals) > 1e-12
        V.fill(np.nan + 0j)
        np.divide(left_vals, right_vals, out=V, where=valid)
        self.last_V = V.copy()

        A = np.asarray(self.calibration.A, dtype=np.complex128)
        B = np.asarray(self.calibration.B, dtype=np.complex128)
        C = np.asarray(self.calibration.C, dtype=np.complex128)
        D = np.asarray(self.calibration.D, dtype=np.complex128)

        if A.size == 1:
            A = np.full(V.shape, A[0], dtype=np.complex128)
        if B.size == 1:
            B = np.full(V.shape, B[0], dtype=np.complex128)
        if C.size == 1:
            C = np.full(V.shape, C[0], dtype=np.complex128)
        if D.size == 1:
            D = np.full(V.shape, D[0], dtype=np.complex128)

        Z = (A * V + B) / (C * V + D)

        magnitude_db = np.full(V.shape, -120.0, dtype=np.float64)
        phase_deg = np.zeros(V.shape, dtype=np.float64)
        valid_Z = np.isfinite(Z)
        magnitude_db[valid_Z] = 20.0 * np.log10(np.abs(Z[valid_Z]) + 1e-12)
        phase_deg[valid_Z] = np.degrees(np.angle(Z[valid_Z]))

        meas_db = self.compute_pp_dbfs(left_data[:, 0])
        ref_db = self.compute_pp_dbfs(right_data[:, 0])
        self.fft_count += 1

        self.spectrum_ready.emit({
            "frequencies": frequencies,
            "magnitude_db": magnitude_db,
            "phase_deg": phase_deg,
            "ref_dbfs": ref_db,
            "meas_dbfs": meas_db,
            "abs_Z": np.abs(Z),
            "fft_count": self.fft_count,
        })

    def run_loop(self):
        if self.running:
            self.process_fft_block()


# -----------------------------------------------------------------------------
# Plot canvas
# -----------------------------------------------------------------------------

class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5.5), dpi=100)
        self.ax_mag = self.fig.add_subplot(111)
        self.ax_mag.set_xscale("log")
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True, which="both", alpha=0.3)

        self.ax_phase = self.ax_mag.twinx()
        self.ax_phase.set_ylabel("Phase (°)")
        self.ax_phase.set_ylim(-90, 90)

        self.line_mag, = self.ax_mag.plot([], [], "b-", lw=2)
        self.line_phase, = self.ax_phase.plot([], [], "r-", lw=2)

        super().__init__(self.fig)
        self.setParent(parent)

    def update_plot(self, frequencies, magnitude_values, phase_values):
        self.line_mag.set_data(frequencies, magnitude_values)
        self.line_phase.set_data(frequencies, phase_values)
        self.ax_mag.relim()
        self.ax_mag.autoscale_view(scalex=True, scaley=True)
        self.fig.canvas.draw_idle()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class AllowedFFTSizeSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(256, 262144)
        self.setSingleStep(1)
        self.setValue(FFT_SIZE)

    def stepBy(self, steps: int):
        current = self.value()
        if steps > 0:
            nxt = current
            while True:
                nxt += 1
                if nxt > 262144:
                    nxt = 262144
                    break
                val = normalize_fft_size(nxt)
                if val > current:
                    break
            self.setValue(normalize_fft_size(nxt))
        else:
            nxt = current
            while True:
                nxt -= 1
                if nxt < 256:
                    nxt = 256
                    break
                val = normalize_fft_size(nxt)
                if val < current:
                    break
            self.setValue(normalize_fft_size(nxt))


class ImpedanceAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 900)

        self.sample_rate = SAMPLE_RATE
        self.fft_size = FFT_SIZE
        self.reference_resistance = REF_RESISTANCE
        self.sparse_bins = compute_sparse_frequencies(self.sample_rate, self.fft_size)

        self.audio_device_id = self.select_default_device_idx()
        self.audio_device_name = self.find_device_name(self.audio_device_id)

        self.audio_manager = AudioStreamManager(self.audio_device_id, self.sample_rate, self.fft_size, self.sparse_bins)
        self.dsp_worker = DSPWorker(self.audio_manager, self.sparse_bins)

        self.build_menu_bar()

        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.canvas = SpectrumCanvas(self)
        layout.addWidget(self.canvas, 3)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel, 1)

        # FFT size
        fft_row = QHBoxLayout()
        fft_label = QLabel("FFT size:")
        self.fft_size_spinbox = AllowedFFTSizeSpinBox(self)
        self.fft_size_spinbox.setValue(self.fft_size)
        self.fft_size_spinbox.setToolTip("Press Enter or click Apply to accept the value.")
        self.fft_size_apply = QPushButton("Apply")
        self.fft_size_apply.clicked.connect(self.on_fft_size_apply)
        self.fft_size_spinbox.editingFinished.connect(self.on_fft_size_enter)
        fft_row.addWidget(fft_label)
        fft_row.addWidget(self.fft_size_spinbox)
        fft_row.addWidget(self.fft_size_apply)
        right_layout.addLayout(fft_row)

        # Frequency points
        freq_label = QLabel(f"Frequency points: {len(self.sparse_bins)}")
        right_layout.addWidget(freq_label)

        # Reference resistance
        self.ref_res_spinbox = QSpinBox()
        self.ref_res_spinbox.setRange(1, int(1e6))
        self.ref_res_spinbox.setValue(int(self.reference_resistance))
        self.ref_res_spinbox.valueChanged.connect(self.on_reference_resistance_changed)
        right_layout.addWidget(QLabel("Reference resistance (Ohm):"))
        right_layout.addWidget(self.ref_res_spinbox)

        right_layout.addWidget(QLabel("Reference level:"))
        self.ref_level_bar = QProgressBar()
        self.ref_level_bar.setRange(0, 100)
        self.ref_level_bar.setValue(0)
        self.ref_level_bar.setFormat("-60.0 dBFS")
        right_layout.addWidget(self.ref_level_bar)

        right_layout.addWidget(QLabel("Measurement level:"))
        self.meas_level_bar = QProgressBar()
        self.meas_level_bar.setRange(0, 100)
        self.meas_level_bar.setValue(0)
        self.meas_level_bar.setFormat("-60.0 dBFS")
        right_layout.addWidget(self.meas_level_bar)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)

        # Calibration controls
        right_layout.addWidget(QLabel("Calibration:"))

        self.cal_open_btn = QPushButton("Calibrate Open")
        self.cal_short_btn = QPushButton("Calibrate Short")
        self.cal_load_btn = QPushButton("Calibrate Load")
        self.apply_cal_btn = QPushButton("Apply Cal")
        self.ratio_only_btn = QPushButton("Ratio Only")

        for btn in [self.cal_open_btn, self.cal_short_btn, self.cal_load_btn, self.apply_cal_btn, self.ratio_only_btn]:
            btn.setStyleSheet("background-color: lightgray;")

        self.cal_open_btn.clicked.connect(self.on_cal_open)
        self.cal_short_btn.clicked.connect(self.on_cal_short)
        self.cal_load_btn.clicked.connect(self.on_cal_load)
        self.apply_cal_btn.clicked.connect(self.on_apply_cal)
        self.ratio_only_btn.clicked.connect(self.on_ratio_only)

        right_layout.addWidget(self.cal_open_btn)
        right_layout.addWidget(self.cal_short_btn)
        right_layout.addWidget(self.cal_load_btn)
        right_layout.addWidget(self.apply_cal_btn)
        right_layout.addWidget(self.ratio_only_btn)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        self.dsp_worker.spectrum_ready.connect(self.on_spectrum_ready)
        self.dsp_timer = QTimer(self)
        self.dsp_timer.timeout.connect(self.dsp_worker.run_loop)
        self.dsp_timer.setInterval(50)

        self.V_open = None
        self.V_short = None
        self.V_load = None
        self.apply_ratio_only()

    def build_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self.action_load_cfg = QAction("Load config & cal", self)
        self.action_save_cfg = QAction("Save config & cal as...", self)
        self.action_exit = QAction("Exit", self)
        self.action_load_cfg.triggered.connect(self.on_load_config)
        self.action_save_cfg.triggered.connect(self.on_save_config)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_load_cfg)
        file_menu.addAction(self.action_save_cfg)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        sound_menu = menubar.addMenu("Soundcard")
        self.soundcard_combo = QComboBox(self)
        self.populate_soundcard_menu()
        combo_action = QWidgetAction(self)
        combo_action.setDefaultWidget(self.soundcard_combo)
        sound_menu.addAction(combo_action)

        about_menu = menubar.addMenu("About")
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self.on_about)
        about_menu.addAction(self.action_about)

    def populate_soundcard_menu(self):
        self.soundcard_combo.clear()
        devices = get_soundcard_devices()
        if not devices:
            self.soundcard_combo.addItem("No compatible soundcards found")
            self.soundcard_combo.setEnabled(False)
            return

        self._soundcards = devices
        for idx, name in devices:
            self.soundcard_combo.addItem(name)
        selected_index = self.current_device_index_in_list(self.audio_device_id)
        if selected_index is not None:
            self.soundcard_combo.setCurrentIndex(selected_index)
        else:
            self.soundcard_combo.setCurrentIndex(0)
        self.soundcard_combo.currentIndexChanged.connect(self.on_soundcard_changed)

    def current_device_index_in_list(self, device_id: int) -> Optional[int]:
        for idx, (did, _) in enumerate(self._soundcards):
            if did == device_id:
                return idx
        return None

    def find_device_name(self, device_id: int) -> str:
        try:
            info = sd.query_devices(device_id)
            if isinstance(info, dict):
                return str(info.get("name", f"device_{device_id}"))
        except Exception:
            pass
        return f"device_{device_id}"

    def select_default_device_idx(self) -> int:
        devices = get_soundcard_devices()
        if not devices:
            return int(sd.default.device)
        return devices[0][0]

    def on_about(self):
        QMessageBox.information(self, "About", f"{APP_TITLE}\n\nA two-channel impedance spectrograph.")

    def on_soundcard_changed(self, idx: int):
        if not hasattr(self, "_soundcards"):
            return
        if idx < 0 or idx >= len(self._soundcards):
            return
        selected_id, selected_name = self._soundcards[idx]
        self.audio_device_id = selected_id
        self.audio_device_name = selected_name
        self.status_label.setText(f"Soundcard selected: {selected_name}")

    def on_reference_resistance_changed(self, value: int):
        self.reference_resistance = float(value)

    def on_fft_size_enter(self):
        self.apply_fft_size()

    def on_fft_size_apply(self):
        self.apply_fft_size()

    def apply_fft_size(self):
        value = int(self.fft_size_spinbox.value())
        valid = normalize_fft_size(value)
        self.fft_size_spinbox.setValue(valid)
        self.fft_size = valid
        self.sample_rate = self.sample_rate
        self.sparse_bins = compute_sparse_frequencies(self.sample_rate, self.fft_size)
        self.status_label.setText(f"FFT size set to {self.fft_size}; {len(self.sparse_bins)} tones.")

    def on_start(self):
        self.dsp_worker.running = True
        self.dsp_timer.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Acquisition running.")

    def on_stop(self):
        self.dsp_worker.running = False
        self.dsp_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Acquisition stopped.")

    def set_button_state(self, button: QPushButton, active: bool):
        button.setStyleSheet("background-color: lightgreen;" if active else "background-color: lightgray;")

    def set_calibration_buttons_gray(self):
        for btn in [self.cal_open_btn, self.cal_short_btn, self.cal_load_btn, self.apply_cal_btn, self.ratio_only_btn]:
            btn.setStyleSheet("background-color: lightgray;")

    def on_cal_open(self):
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet.")
            return
        self.V_open = self.dsp_worker.last_V.copy()
        self.set_button_state(self.cal_open_btn, True)
        self.status_label.setText("Open calibration captured.")

    def on_cal_short(self):
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet.")
            return
        self.V_short = self.dsp_worker.last_V.copy()
        self.set_button_state(self.cal_short_btn, True)
        self.status_label.setText("Short calibration captured.")

    def on_cal_load(self):
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet.")
            return
        self.V_load = self.dsp_worker.last_V.copy()
        self.set_button_state(self.cal_load_btn, True)
        self.status_label.setText("Load calibration captured.")

    def apply_saved_calibration(self):
        if self.V_open is None or self.V_short is None or self.V_load is None:
            self.status_label.setText("Calibration incomplete: Open, Short, and Load are all required.")
            return
        try:
            coeffs = apply_calibration_coefficients(self.V_open, self.V_short, self.V_load, self.reference_resistance)
            self.dsp_worker.calibration = coeffs
            self.status_label.setText("Calibration applied successfully.")
        except ValueError as exc:
            self.status_label.setText(f"Calibration error: {exc}")

    def on_apply_cal(self):
        self.set_calibration_buttons_gray()
        self.apply_saved_calibration()
        self.set_button_state(self.apply_cal_btn, True)
        self.set_button_state(self.cal_open_btn, self.V_open is not None)
        self.set_button_state(self.cal_short_btn, self.V_short is not None)
        self.set_button_state(self.cal_load_btn, self.V_load is not None)

    def apply_ratio_only(self):
        self.set_calibration_buttons_gray()
        self.V_open = None
        self.V_short = None
        self.V_load = None
        self.dsp_worker.calibration = default_calibration()
        self.set_button_state(self.ratio_only_btn, True)
        self.canvas.ax_mag.set_ylabel("Magnitude (dB)")
        self.status_label.setText("Ratio-only mode active.")

    def on_ratio_only(self):
        self.apply_ratio_only()

    @Slot(dict)
    def on_spectrum_ready(self, data: dict):
        frequencies = data["frequencies"]
        mag_db = data["magnitude_db"]
        phase_deg = data["phase_deg"]
        ref_db = data["ref_dbfs"]
        meas_db = data["meas_dbfs"]

        if self.dsp_worker.calibration.A.size == 1 and self.dsp_worker.calibration.A[0] == 1.0 and self.dsp_worker.calibration.B.size == 1 and self.dsp_worker.calibration.B[0] == 0.0 and self.dsp_worker.calibration.C.size == 1 and self.dsp_worker.calibration.C[0] == 0.0 and self.dsp_worker.calibration.D.size == 1 and self.dsp_worker.calibration.D[0] == 1.0:
            self.canvas.ax_mag.set_ylabel("Magnitude (dB)")
            self.canvas.ax_mag.set_yscale("linear")
            self.canvas.ax_mag.set_ylim(None, None)
            self.canvas.update_plot(frequencies, mag_db, phase_deg)
        else:
            abs_z = np.abs(data["abs_Z"])
            ref_min = max(0.01 * self.reference_resistance, 1e-6)
            ref_max = 100.0 * self.reference_resistance
            abs_min = float(np.min(abs_z[np.isfinite(abs_z)])) if np.any(np.isfinite(abs_z)) else ref_min
            abs_max = float(np.max(abs_z[np.isfinite(abs_z)])) if np.any(np.isfinite(abs_z)) else ref_max
            y_min = max(ref_min, 0.83 * abs_min)
            y_max = max(ref_max, 1.20 * abs_max)
            y_min = min(y_min, y_max / 10.0)
            y_max = max(y_max, y_min * 1.2)
            self.canvas.ax_mag.set_ylabel("Magnitude (Ohm)")
            self.canvas.ax_mag.set_yscale("log")
            self.canvas.ax_mag.set_ylim(y_min, y_max)
            self.canvas.update_plot(frequencies, abs_z, phase_deg)

        self.canvas.ax_phase.set_ylim(-90, 90)
        self.ref_level_bar.setValue(int(np.clip((ref_db + 6.0) / 6.0 * 100.0, 0, 100)))
        self.ref_level_bar.setFormat(f"{ref_db:.1f} dBFS")
        self.meas_level_bar.setValue(int(np.clip((meas_db + 6.0) / 6.0 * 100.0, 0, 100)))
        self.meas_level_bar.setFormat(f"{meas_db:.1f} dBFS")

        self.status_label.setText(f"FFT #{data['fft_count']}: {len(frequencies)} tones processed.")

    def on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config & cal", "", "JSON files (*.json)")
        if not path:
            return
        payload = self.build_config_dict()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.status_label.setText(f"Saved config to {path}")

    def build_config_dict(self) -> dict[str, Any]:
        coeffs = []
        if self.dsp_worker.calibration.A is not None and np.asarray(self.dsp_worker.calibration.A).size == len(self.sparse_bins):
            A = np.asarray(self.dsp_worker.calibration.A, dtype=np.complex128)
            B = np.asarray(self.dsp_worker.calibration.B, dtype=np.complex128)
            C = np.asarray(self.dsp_worker.calibration.C, dtype=np.complex128)
            D = np.asarray(self.dsp_worker.calibration.D, dtype=np.complex128)
            for i in range(len(self.sparse_bins)):
                coeffs.append([A[i].real, A[i].imag, B[i].real, B[i].imag, C[i].real, C[i].imag, D[i].real, D[i].imag])
        else:
            coeffs = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] for _ in self.sparse_bins]

        return {
            "application": APP_TITLE,
            "soundcard_name": self.audio_device_name,
            "sample_rate": int(self.sample_rate),
            "fft_size": int(self.fft_size),
            "bins": [int(v) for v in self.sparse_bins],
            "reference_resistance": float(self.reference_resistance),
            "coefficients": coeffs,
        }

    def on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config & cal", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            self.load_config_dict(obj)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load config: {exc}")
            self.status_label.setText("Failed to load config.")

    def load_config_dict(self, obj: dict[str, Any]):
        if not isinstance(obj, dict):
            raise ValueError("Config file format is invalid.")

        # soundcard
        soundcard_name = str(obj.get("soundcard_name", ""))
        devices = get_soundcard_devices()
        self.audio_device_id = self.select_default_device_idx()
        for idx, name in devices:
            if name == soundcard_name:
                self.audio_device_id = idx
                self.audio_device_name = name
                break
        else:
            self.audio_device_name = self.find_device_name(self.audio_device_id)

        # sample rate
        sample_rate = int(obj.get("sample_rate", SAMPLE_RATE))
        if not (32000 <= sample_rate <= 384000):
            sample_rate = SAMPLE_RATE
        if self.audio_device_id is not None:
            try:
                info = sd.query_devices(self.audio_device_id)
                if isinstance(info, dict):
                    max_sr = int(info.get("default_samplerate", 48000) or 48000)
                    if sample_rate > max_sr:
                        sample_rate = max_sr
            except Exception:
                pass
        self.sample_rate = sample_rate

        # fft size
        fft_size = int(obj.get("fft_size", FFT_SIZE))
        self.fft_size = normalize_fft_size(fft_size)

        # bins
        bins = list(obj.get("bins", self.sparse_bins))
        bins = sanitize_bins(bins, self.fft_size)
        if not (2 <= len(bins) <= 400):
            bins = compute_sparse_frequencies(self.sample_rate, self.fft_size)
        self.sparse_bins = bins

        # reference resistance
        ref_res = float(obj.get("reference_resistance", REF_RESISTANCE))
        if not (1.0 <= ref_res <= 1.0e6):
            ref_res = REF_RESISTANCE
        self.reference_resistance = ref_res
        self.ref_res_spinbox.setValue(int(ref_res))

        # calibration coefficients
        cal_list = obj.get("coefficients", [])
        A = np.zeros(len(self.sparse_bins), dtype=np.complex128)
        B = np.zeros(len(self.sparse_bins), dtype=np.complex128)
        C = np.zeros(len(self.sparse_bins), dtype=np.complex128)
        D = np.ones(len(self.sparse_bins), dtype=np.complex128)

        if isinstance(cal_list, list) and len(cal_list) >= len(self.sparse_bins):
            for i in range(len(self.sparse_bins)):
                row = cal_list[i]
                if not isinstance(row, (list, tuple)) or len(row) < 8:
                    A[i], B[i], C[i], D[i] = 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j
                    continue
                A[i], B[i], C[i], D[i] = normalize_calibration_coeffs_single_point(
                    complex(row[0], row[1]),
                    complex(row[2], row[3]),
                    complex(row[4], row[5]),
                    complex(row[6], row[7]),
                )
        self.dsp_worker.calibration = CalibrationCoefficients(A=A, B=B, C=C, D=D)

        # Stop and restart stream if necessary
        self.audio_manager.stop()
        self.audio_manager = AudioStreamManager(self.audio_device_id, self.sample_rate, self.fft_size, self.sparse_bins)
        self.dsp_worker = DSPWorker(self.audio_manager, self.sparse_bins)
        self.dsp_worker.spectrum_ready.connect(self.on_spectrum_ready)
        self.audio_manager.start()
        self.dsp_worker.running = True
        self.dsp_timer.start()
        self.status_label.setText("Loaded config and calibration. Audio restarted.")


def main():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if args.list_devices:
        devices = get_soundcard_devices()
        print("Compatible devices:")
        if not devices:
            print("No compatible device found.")
        else:
            for idx, name in devices:
                print(f"{idx}: {name}")
        return

    app = QApplication(sys.argv)
    window = ImpedanceAnalyzerWindow()
    if args.device is not None:
        window.audio_device_id = args.device
        window.audio_device_name = window.find_device_name(args.device)
        window.soundcard_combo.setCurrentIndex(window.current_device_index_in_list(args.device) or 0)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
