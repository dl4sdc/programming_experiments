#!/usr/bin/env python3
"""
Impedance Spectrum Analyzer v0.5 (matplotlib backend)
Uses a sound card (PortAudio) for duplex audio I/O.
Transmits a sparse multi-tone test signal and measures impedance via two-channel FFT analysis.

Architecture:
  - Main thread: Qt event loop, UI updates via signals
  - Audio thread: sounddevice callback manages ring buffers (send/receive)
  - DSP worker: calculates FFTs, complex quotients, impedance using numpy vectorization
  - Plotting: matplotlib with PySide6 backend for reliable dual y-axis with log x-scale
  - Calibration: three-step (open/short/load) with numpy array operations

Usage:
  python impedance_meas.py                    # Use default soundcard
  python impedance_meas.py --list-devices     # List available soundcards
  python impedance_meas.py --device <id>      # Select soundcard by index
"""

import argparse
import cmath
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QObject
from PySide6.QtGui import QColor, QPen, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QComboBox,
    QSlider,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib


# ============================================================================
# Constants and Configuration
# ============================================================================

SAMPLE_RATE = 48000
FFT_SIZE = 48000
BUFFER_SEND_LEN = 2 * FFT_SIZE
BUFFER_RECV_LEN = 3 * FFT_SIZE

DAC_SPAN_PEAK = 32767.0
TEST_SIGNAL_AMPLITUDE = (120.0 / 128.0) * DAC_SPAN_PEAK
ADC_FULL_SCALE_PP = 2.0 * 32767.0

FREQ_RATIO_MAX = 1.0594631  # semitone
FREQ_START_FRACTION = 5.0 / 12.0  # 5/12 of sample rate
REF_RESISTANCE = 100.0


@dataclass
class CalibrationCoefficients:
    """Complex calibration coefficients for two-port impedance calculation."""
    A: np.ndarray | complex = None
    B: np.ndarray | complex = None
    C: np.ndarray | complex = None
    D: np.ndarray | complex = None

    def __post_init__(self):
        """Ensure all coefficients are initialized with proper defaults."""
        if self.A is None:
            self.A = np.ones(1, dtype=np.complex128)
        if self.B is None:
            self.B = np.zeros(1, dtype=np.complex128)
        if self.C is None:
            self.C = np.zeros(1, dtype=np.complex128)
        if self.D is None:
            self.D = np.ones(1, dtype=np.complex128)


# ============================================================================
# Sparse Frequency Computation (Computed Once)
# ============================================================================

def compute_sparse_frequencies() -> list[int]:
    """
    Compute sparse frequency points (FFT bin indices) such that consecutive
    frequencies have a maximum ratio of FREQ_RATIO_MAX, starting from
    5/12 of the sample rate and stepping down.

    Skips 50 Hz to avoid line frequency interference.
    Ensures ratio of adjacent tones <= FREQ_RATIO_MAX.

    Returns:
        List of FFT bin indices in ascending order.
    """
    start_freq = FREQ_START_FRACTION * SAMPLE_RATE
    start_bin = int(start_freq / (SAMPLE_RATE / FFT_SIZE))

    bins: list[int] = []
    current_bin = start_bin
    while current_bin >= 1:
        bins.append(current_bin)
        next_bin = int(np.ceil(current_bin / FREQ_RATIO_MAX))
        if next_bin == current_bin or next_bin < 10:
            break
        if next_bin == 50:
            if current_bin == 51:
                next_bin = 49
            else:
                next_bin = 51
        current_bin = next_bin

    return sorted(bins)


SPARSE_FREQ_BINS = compute_sparse_frequencies()


def schroeder_phase_sequence(n: int) -> np.ndarray:
    """
    Generate Schroeder phase sequence for n tones to minimize crest factor.
    Phases are in radians.
    """
    return np.array([
        -np.pi * k * (k + 1) / n for k in range(n)
    ])


def synthesize_multitone_fft(fft_size: int, bins: list[int]) -> np.ndarray:
    """
    Synthesize a multi-tone test signal in the frequency domain.
    Returns the FFT spectrum (complex array of size fft_size).
    """
    spectrum = np.zeros(fft_size, dtype=np.complex128)
    phases = schroeder_phase_sequence(len(bins))

    for k, bin_idx in enumerate(bins):
        phase = phases[k]
        amplitude = TEST_SIGNAL_AMPLITUDE / np.sqrt(len(bins))
        spectrum[bin_idx] = amplitude * np.exp(1j * phase)
        if bin_idx != 0 and bin_idx != fft_size // 2:
            mirror = fft_size - bin_idx
            spectrum[mirror] = amplitude * np.exp(-1j * phase)

    return spectrum


def generate_multitone_time_signal(sample_count: int, bins: list[int]) -> np.ndarray:
    """
    Generate time-domain multi-tone test signal by inverse FFT.
    Pads to sample_count with zeros at the end.
    Returns real-valued time-domain signal (int16 normalized to DAC span).
    """
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

    return time_signal.astype(np.int16)


def apply_calibration_coefficients(v_open: np.ndarray, v_short: np.ndarray,
                                    v_load: np.ndarray, ref_resistance: float) -> CalibrationCoefficients:
    """
    Calculate calibration arrays for the two-port equation:
        Z = (A*V + B) / (C*V + D)

    With D = 1.0 for the over-determined system.
    Uses numpy vectorization for all array operations.
    """
    v_open = np.asarray(v_open, dtype=np.complex128)
    v_short = np.asarray(v_short, dtype=np.complex128)
    v_load = np.asarray(v_load, dtype=np.complex128)

    open_valid = np.abs(v_open) > 1e-12
    if not np.all(open_valid):
        raise ValueError(
            f"Open calibration has near-zero voltages at {np.sum(~open_valid)} frequencies. "
            "Check measurement."
        )

    denom = v_load - v_short
    denom_valid = np.abs(denom) > 1e-12
    if not np.all(denom_valid):
        raise ValueError(
            f"Load and short calibration values are too similar at {np.sum(~denom_valid)} frequencies."
        )

    C = np.zeros_like(v_open, dtype=np.complex128)
    A = np.zeros_like(v_open, dtype=np.complex128)
    B = np.zeros_like(v_open, dtype=np.complex128)
    D = np.ones_like(v_open, dtype=np.complex128)

    valid = open_valid & denom_valid
    C[valid] = -1.0 / v_open[valid]
    A[valid] = ref_resistance * (1.0 - v_load[valid] / v_open[valid]) / denom[valid]
    B[valid] = -A[valid] * v_short[valid]

    return CalibrationCoefficients(A=A, B=B, C=C, D=D)


# ============================================================================
# Ring Buffer Management
# ============================================================================

class RingBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(self, capacity: int, num_channels: int = 1):
        self.capacity = capacity
        self.num_channels = num_channels
        self.buffer = np.zeros((capacity, num_channels), dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Write data to the buffer, wrapping around if necessary."""
        with self.lock:
            samples = len(data)
            space_to_end = self.capacity - self.write_pos
            if samples <= space_to_end:
                self.buffer[self.write_pos : self.write_pos + samples] = data
            else:
                self.buffer[self.write_pos :] = data[:space_to_end]
                self.buffer[: samples - space_to_end] = data[space_to_end:]
            self.write_pos = (self.write_pos + samples) % self.capacity

    def read(self, start_pos: int, length: int) -> np.ndarray:
        """Read data starting from start_pos with circular wrap."""
        with self.lock:
            if length == 0:
                return np.array([], dtype=np.float32).reshape(0, self.num_channels)
            pos = start_pos % self.capacity
            end_pos = (pos + length) % self.capacity
            if pos < end_pos:
                return self.buffer[pos:end_pos].copy()
            else:
                return np.vstack([self.buffer[pos:], self.buffer[:end_pos]]).copy()

    def get_write_pos(self) -> int:
        """Return current write position."""
        with self.lock:
            return self.write_pos


# ============================================================================
# Audio Callback and Stream Management
# ============================================================================

class AudioStreamManager:
    """Manages PortAudio stream with duplex I/O and ring buffers."""

    def __init__(self, device_id: int, sample_rate: int, fft_size: int, sparse_bins: list[int]):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.sparse_bins = list(sparse_bins)
        self.stream = None

        self.send_buffer = RingBuffer(BUFFER_SEND_LEN, num_channels=1)
        self.recv_left = RingBuffer(BUFFER_RECV_LEN, num_channels=1)
        self.recv_right = RingBuffer(BUFFER_RECV_LEN, num_channels=1)

        test_signal = generate_multitone_time_signal(fft_size, self.sparse_bins)
        self.send_buffer.write(test_signal.astype(np.float32).reshape(-1, 1) / DAC_SPAN_PEAK)
        self.send_buffer.write(test_signal.astype(np.float32).reshape(-1, 1) / DAC_SPAN_PEAK)

        self.recv_write_pos = 0
        self.dsp_read_pos = 0
        self.total_samples = 0
        self.lock = threading.Lock()

        self.callback_count = 0
        self.underrun_count = 0

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int,
                       time_info, status):
        """
        PortAudio callback for duplex I/O.
        Consumes from send_buffer, accumulates to recv_left/recv_right.
        """
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        send_data = self.send_buffer.read(self.total_samples % BUFFER_SEND_LEN, frames)
        outdata[:, 0] = send_data[:, 0]
        outdata[:, 1] = send_data[:, 0]

        with self.lock:
            self.recv_left.write(indata[:, 0].reshape(-1, 1))
            self.recv_right.write(indata[:, 1].reshape(-1, 1))
            self.recv_write_pos = self.recv_left.get_write_pos()
            self.total_samples += frames
            self.callback_count += 1

    def start(self):
        """Start the audio stream."""
        try:
            self.stream = sd.Stream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=2,
                blocksize=4096,
                callback=self.audio_callback,
                latency="low",
            )
            self.stream.start()
        except Exception as e:
            print(f"Failed to start audio stream: {e}", file=sys.stderr)
            raise

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ============================================================================
# DSP Worker Thread
# ============================================================================

class DSPWorker(QObject):
    """
    Processes audio data: FFT calculation, complex quotients, impedance.
    Runs in a separate QThread and emits signals for UI updates.
    Uses numpy vectorization for all calculations.
    """

    spectrum_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, stream_manager: AudioStreamManager, sparse_bins: list[int]):
        super().__init__()
        self.stream_manager = stream_manager
        self.running = False
        self.fft_size = stream_manager.fft_size
        self.sample_rate = stream_manager.sample_rate
        self.sparse_bins = list(sparse_bins)
        self.sparse_bins_array = np.array(self.sparse_bins, dtype=np.int64)

        self.calibration = CalibrationCoefficients()
        self.last_V = None
        self.last_ref_pp_dbfs = -60.0
        self.last_meas_pp_dbfs = -60.0
        self.last_recv_pos = 0
        self.fft_count = 0

    @staticmethod
    def compute_pp_dbfs(data: np.ndarray) -> float:
        """
        Compute peak-to-peak amplitude in dBFS.
        Clamps to -6 dB to 0 dB range for bargraph display.
        """
        pp = float(np.ptp(data))
        if pp <= 0.0:
            return -60.0
        dbfs = 20.0 * np.log10(pp / ADC_FULL_SCALE_PP)
        return float(np.clip(dbfs, -6.0, 0.0))

    def process_fft_block(self):
        """
        If a complete FFT_SIZE block is available, process it using numpy vectorization.
        """
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

        valid = np.abs(right_vals) > 1e-9
        V = np.empty(self.sparse_bins_array.size, dtype=np.complex128)
        V.fill(np.nan + 0j)
        np.divide(left_vals, right_vals, out=V, where=valid)
        self.last_V = V.copy()

        A = np.asarray(self.calibration.A, dtype=np.complex128)
        B = np.asarray(self.calibration.B, dtype=np.complex128)
        C = np.asarray(self.calibration.C, dtype=np.complex128)
        D = np.asarray(self.calibration.D, dtype=np.complex128)

        if A.size == 1:
            A = np.ones_like(V, dtype=np.complex128) * A[0]
        if B.size == 1:
            B = np.zeros_like(V, dtype=np.complex128) + B[0]
        if C.size == 1:
            C = np.zeros_like(V, dtype=np.complex128) + C[0]
        if D.size == 1:
            D = np.ones_like(V, dtype=np.complex128) * D[0]

        Z = (A * V + B) / (C * V + D)

        magnitude_db = np.full(V.shape, -120.0, dtype=np.float64)
        phase_deg = np.zeros(V.shape, dtype=np.float64)

        valid_z = np.isfinite(Z)
        magnitude_db[valid_z] = 20.0 * np.log10(np.abs(Z[valid_z]) + 1e-9)
        phase_deg[valid_z] = np.degrees(np.angle(Z[valid_z]))

        self.last_meas_pp_dbfs = self.compute_pp_dbfs(left_data[:, 0])
        self.last_ref_pp_dbfs = self.compute_pp_dbfs(right_data[:, 0])

        self.fft_count += 1

        self.spectrum_ready.emit({
            'magnitude_db': magnitude_db,
            'phase_deg': phase_deg,
            'frequencies': frequencies,
            'fft_count': self.fft_count,
            'ref_dbfs': self.last_ref_pp_dbfs,
            'meas_dbfs': self.last_meas_pp_dbfs,
        })

    def run_loop(self):
        """Main processing loop (called by Qt timer from main thread)."""
        if self.running:
            self.process_fft_block()


# ============================================================================
# Matplotlib Canvas for Dual Axis Plot
# ============================================================================

class SpectrumCanvas(FigureCanvas):
    """Matplotlib canvas with dual y-axes for magnitude and phase."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('white')

        self.ax_mag = self.fig.add_subplot(111)
        self.ax_mag.set_xlabel('Frequency (Hz)', fontsize=10)
        self.ax_mag.set_ylabel('Magnitude (dB)', color='blue', fontsize=10)
        self.ax_mag.tick_params(axis='y', labelcolor='blue')
        self.ax_mag.set_xscale('log')
        self.ax_mag.grid(True, alpha=0.3, which='both')

        self.ax_phase = self.ax_mag.twinx()
        self.ax_phase.set_ylabel('Phase (°)', color='red', fontsize=10)
        self.ax_phase.tick_params(axis='y', labelcolor='red')
        self.ax_phase.set_ylim(-180, 180)

        self.line_mag, = self.ax_mag.plot([], [], 'b-', linewidth=2, label='Magnitude', zorder=2)
        self.line_phase, = self.ax_phase.plot([], [], 'r-', linewidth=2, label='Phase', zorder=1)

        lines_mag = [self.line_mag]
        lines_phase = [self.line_phase]
        labels_mag = [l.get_label() for l in lines_mag]
        labels_phase = [l.get_label() for l in lines_phase]

        self.ax_mag.legend(lines_mag, labels_mag, loc='upper left', fontsize=9)
        self.ax_phase.legend(lines_phase, labels_phase, loc='upper right', fontsize=9)

        self.fig.tight_layout()

        super().__init__(self.fig)
        self.setParent(parent)

    def update_plot(self, frequencies, magnitude_db, phase_deg):
        """Update plot with new data."""
        self.line_mag.set_data(frequencies, magnitude_db)
        self.line_phase.set_data(frequencies, phase_deg)

        self.ax_mag.relim()
        self.ax_mag.autoscale_view(scalex=True, scaley=True)

        self.fig.canvas.draw_idle()


# ============================================================================
# Qt UI Main Window
# ============================================================================

class ImpedanceAnalyzerUI(QMainWindow):
    """Main application window with controls, plots, and status."""

    def __init__(self, stream_manager: AudioStreamManager, dsp_worker: DSPWorker):
        super().__init__()
        self.stream_manager = stream_manager
        self.dsp_worker = dsp_worker

        self.setWindowTitle("Impedance Spectrum Analyzer v0.5 (matplotlib + calibration)")
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.canvas = SpectrumCanvas(self)
        main_layout.addWidget(self.canvas, 3)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_spinbox = QSpinBox()
        self.fft_size_spinbox.setMinimum(1024)
        self.fft_size_spinbox.setMaximum(262144)
        self.fft_size_spinbox.setValue(FFT_SIZE)
        self.fft_size_spinbox.setEnabled(False)
        control_layout.addWidget(self.fft_size_spinbox)

        control_layout.addWidget(QLabel("Frequency Points:"))
        self.freq_points_label = QLabel(f"{len(self.dsp_worker.sparse_bins)}")
        control_layout.addWidget(self.freq_points_label)

        control_layout.addWidget(QLabel("Reference Resistance (Ω):"))
        self.ref_resistance_spinbox = QDoubleSpinBox()
        self.ref_resistance_spinbox.setMinimum(1.0)
        self.ref_resistance_spinbox.setMaximum(1e6)
        self.ref_resistance_spinbox.setDecimals(1)
        self.ref_resistance_spinbox.setValue(REF_RESISTANCE)
        control_layout.addWidget(self.ref_resistance_spinbox)

        control_layout.addWidget(QLabel("Reference Level:"))
        self.ref_level_bar = QProgressBar()
        self.ref_level_bar.setRange(0, 100)
        self.ref_level_bar.setValue(0)
        self.ref_level_bar.setFormat("-6 dBFS")
        self.ref_level_bar.setStyleSheet("QProgressBar { border: 1px solid gray; text-align: center; }")
        control_layout.addWidget(self.ref_level_bar)

        control_layout.addWidget(QLabel("Measurement Level:"))
        self.meas_level_bar = QProgressBar()
        self.meas_level_bar.setRange(0, 100)
        self.meas_level_bar.setValue(0)
        self.meas_level_bar.setFormat("-6 dBFS")
        self.meas_level_bar.setStyleSheet("QProgressBar { border: 1px solid gray; text-align: center; }")
        control_layout.addWidget(self.meas_level_bar)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.on_start)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        control_layout.addWidget(QLabel("Calibration:"))

        self.cal_open_button = QPushButton("Calibrate Open")
        self.cal_open_button.clicked.connect(self.on_calibrate_open)
        self.cal_open_button.setStyleSheet("background-color: lightgray;")
        control_layout.addWidget(self.cal_open_button)

        self.cal_short_button = QPushButton("Calibrate Short")
        self.cal_short_button.clicked.connect(self.on_calibrate_short)
        self.cal_short_button.setStyleSheet("background-color: lightgray;")
        control_layout.addWidget(self.cal_short_button)

        self.cal_load_button = QPushButton("Calibrate Load")
        self.cal_load_button.clicked.connect(self.on_calibrate_load)
        self.cal_load_button.setStyleSheet("background-color: lightgray;")
        control_layout.addWidget(self.cal_load_button)

        self.apply_cal_button = QPushButton("Apply Cal")
        self.apply_cal_button.clicked.connect(self.on_apply_cal)
        self.apply_cal_button.setStyleSheet("background-color: lightgray;")
        control_layout.addWidget(self.apply_cal_button)

        self.ratio_only_button = QPushButton("Ratio Only")
        self.ratio_only_button.clicked.connect(self.on_ratio_only)
        self.ratio_only_button.setStyleSheet("background-color: lightgray;")
        control_layout.addWidget(self.ratio_only_button)

        control_layout.addStretch()
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        main_layout.addWidget(control_panel, 1)

        self.dsp_worker.spectrum_ready.connect(self.on_spectrum_ready)
        self.dsp_worker.error_occurred.connect(self.on_dsp_error)

        self.dsp_timer = QTimer()
        self.dsp_timer.timeout.connect(self.dsp_worker.run_loop)
        self.dsp_timer.setInterval(50)

        self.V_open = None
        self.V_short = None
        self.V_load = None

        self.on_ratio_only()

    def _set_button_state(self, button: QPushButton, active: bool):
        """Set button background color (green for active, gray for inactive)."""
        color = "lightgreen" if active else "lightgray"
        button.setStyleSheet(f"background-color: {color};")

    def _reset_calibration_buttons(self):
        """Reset all calibration buttons to gray (inactive)."""
        self._set_button_state(self.cal_open_button, False)
        self._set_button_state(self.cal_short_button, False)
        self._set_button_state(self.cal_load_button, False)
        self._set_button_state(self.apply_cal_button, False)
        self._set_button_state(self.ratio_only_button, False)

    def _update_level_bar(self, bar: QProgressBar, value_dbfs: float):
        """Update a level bargraph with dBFS value (-6 to 0 dB range)."""
        value = int(np.clip((value_dbfs + 6.0) / 6.0 * 100.0, 0.0, 100.0))
        bar.setValue(value)
        bar.setFormat(f"{value_dbfs:.1f} dBFS")

    def on_calibrate_open(self):
        """Store last measured V under open condition."""
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet. Start acquisition first.")
            return
        self.V_open = self.dsp_worker.last_V.copy()
        self._set_button_state(self.cal_open_button, True)
        self.status_label.setText("Open calibration saved.")

    def on_calibrate_short(self):
        """Store last measured V under short condition."""
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet. Start acquisition first.")
            return
        self.V_short = self.dsp_worker.last_V.copy()
        self._set_button_state(self.cal_short_button, True)
        self.status_label.setText("Short calibration saved.")

    def on_calibrate_load(self):
        """Store last measured V under resistive load condition."""
        if self.dsp_worker.last_V is None:
            self.status_label.setText("No V data available yet. Start acquisition first.")
            return
        self.V_load = self.dsp_worker.last_V.copy()
        self._set_button_state(self.cal_load_button, True)
        self.status_label.setText("Load calibration saved.")

    def apply_calibration(self):
        """Apply the calibration coefficients using the saved V arrays."""
        if self.V_open is None or self.V_short is None or self.V_load is None:
            self.status_label.setText(
                "Calibration incomplete. All of Open, Short, and Load must be saved first."
            )
            return

        try:
            ref_resistance = float(self.ref_resistance_spinbox.value())
            coeffs = apply_calibration_coefficients(
                self.V_open, self.V_short, self.V_load, ref_resistance
            )
            self.dsp_worker.calibration = coeffs
            self.status_label.setText("Calibration applied successfully.")
        except ValueError as exc:
            self.status_label.setText(f"Calibration error: {exc}")
            print(f"Calibration error: {exc}", file=sys.stderr)

    def on_apply_cal(self):
        """Apply calibration and update button states."""
        self._reset_calibration_buttons()
        self.apply_calibration()
        self._set_button_state(self.apply_cal_button, True)

    def on_ratio_only(self):
        """Return to pure V ratio mode (no calibration)."""
        self._reset_calibration_buttons()
        self.V_open = None
        self.V_short = None
        self.V_load = None

        self.dsp_worker.calibration = CalibrationCoefficients(
            A=np.ones(1, dtype=np.complex128),
            B=np.zeros(1, dtype=np.complex128),
            C=np.zeros(1, dtype=np.complex128),
            D=np.ones(1, dtype=np.complex128),
        )
        self._set_button_state(self.ratio_only_button, True)
        self.status_label.setText("Ratio-only mode (no calibration).")

    @Slot(dict)
    def on_spectrum_ready(self, data: dict):
        """Update plot and level bars when FFT results are ready."""
        frequencies = data['frequencies']
        magnitude_db = data['magnitude_db']
        phase_deg = data['phase_deg']

        self.canvas.update_plot(frequencies, magnitude_db, phase_deg)

        self._update_level_bar(self.ref_level_bar, data['ref_dbfs'])
        self._update_level_bar(self.meas_level_bar, data['meas_dbfs'])

        self.status_label.setText(
            f"FFT #{data['fft_count']}: {len(frequencies)} frequencies, "
            f"freq range {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz"
        )

    @Slot(str)
    def on_dsp_error(self, error_msg: str):
        """Handle DSP worker errors."""
        self.status_label.setText(f"ERROR: {error_msg}")
        print(f"DSP Error: {error_msg}", file=sys.stderr)

    def on_start(self):
        """Start acquisition."""
        self.dsp_worker.running = True
        self.dsp_timer.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Acquisition running...")

    def on_stop(self):
        """Stop acquisition."""
        self.dsp_worker.running = False
        self.dsp_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Acquisition stopped.")

    def closeEvent(self, event):
        """Clean up on window close."""
        self.dsp_timer.stop()
        self.stream_manager.stop()
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Impedance Spectrum Analyzer using sound card and FFT."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio device ID (default: system default).",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return

    app = QApplication(sys.argv)

    device_id = args.device if args.device is not None else sd.default.device

    try:
        print(f"Using device {device_id}: {sd.query_devices(device_id)['name']}")
    except Exception as e:
        print(f"Error querying device: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        stream_manager = AudioStreamManager(device_id, SAMPLE_RATE, FFT_SIZE, SPARSE_FREQ_BINS)
        stream_manager.start()
    except Exception as e:
        print(f"Failed to initialize audio: {e}", file=sys.stderr)
        sys.exit(1)

    dsp_worker = DSPWorker(stream_manager, SPARSE_FREQ_BINS)
    ui = ImpedanceAnalyzerUI(stream_manager, dsp_worker)
    ui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
