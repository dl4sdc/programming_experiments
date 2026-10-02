#!/usr/bin/env python3
"""
Impedance Spectrum Analyzer v0.4 (matplotlib backend)
Uses a sound card (PortAudio) for duplex audio I/O.
Transmits a sparse multi-tone test signal and measures impedance via two-channel FFT analysis.

Architecture:
  - Main thread: Qt event loop, UI updates via signals
  - Audio thread: sounddevice callback manages ring buffers (send/receive)
  - DSP worker: calculates FFTs, complex quotients, impedance
  - Plotting: matplotlib with PySide6 backend for reliable dual y-axis with log x-scale

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

FREQ_RATIO_MAX = 1.0594631  # semitone
FREQ_START_FRACTION = 5.0 / 12.0  # 5/12 of sample rate
REF_RESISTANCE = 100.0


@dataclass
class CalibrationCoefficients:
    """Complex calibration coefficients for two-port impedance calculation."""
    A: complex = complex(REF_RESISTANCE, 0.0)
    B: complex = complex(0.0, 0.0)
    C: complex = complex(-1.0, 0.0)
    D: complex = complex(1.0, 0.0)


# ============================================================================
# Multi-tone Synthesis
# ============================================================================

def compute_sparse_frequencies() -> list[int]:
    """
    Compute sparse frequency points (FFT bin indices) such that consecutive
    frequencies have a maximum ratio of FREQ_RATIO_MAX, starting from
    5/12 of the sample rate and stepping down.

    Returns:
        List of FFT bin indices in ascending order.
    """
    nyquist_bin = FFT_SIZE // 2
    start_freq = FREQ_START_FRACTION * SAMPLE_RATE
    start_bin = int(start_freq / (SAMPLE_RATE / FFT_SIZE))

    bins = []
    current_bin = start_bin
    while current_bin >= 1:
        bins.append(current_bin)
        # Next bin such that ratio of frequencies <= FREQ_RATIO_MAX
        next_bin = int(current_bin / FREQ_RATIO_MAX)
        if next_bin == current_bin or next_bin < 1:
            break
        current_bin = next_bin

    return sorted(bins)


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
        # For conjugate symmetry, set both bin and mirror
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

    # Normalize to DAC span
    peak = np.max(np.abs(time_signal))
    if peak > 0:
        time_signal = time_signal * (TEST_SIGNAL_AMPLITUDE / peak)

    # Pad to requested length
    if sample_count > len(time_signal):
        time_signal = np.pad(time_signal, (0, sample_count - len(time_signal)))
    elif sample_count < len(time_signal):
        time_signal = time_signal[:sample_count]

    return time_signal.astype(np.int16)


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

    def __init__(self, device_id: int, sample_rate: int, fft_size: int):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.stream = None

        # Ring buffers
        self.send_buffer = RingBuffer(BUFFER_SEND_LEN, num_channels=1)
        self.recv_left = RingBuffer(BUFFER_RECV_LEN, num_channels=1)
        self.recv_right = RingBuffer(BUFFER_RECV_LEN, num_channels=1)

        # Fill send buffer with repeating test signal
        test_signal = generate_multitone_time_signal(fft_size, compute_sparse_frequencies())
        self.send_buffer.write(test_signal.astype(np.float32).reshape(-1, 1) / DAC_SPAN_PEAK)
        self.send_buffer.write(test_signal.astype(np.float32).reshape(-1, 1) / DAC_SPAN_PEAK)

        # Tracking
        self.recv_write_pos = 0
        self.dsp_read_pos = 0
        self.total_samples = 0
        self.lock = threading.Lock()

        # Callback statistics
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

        # Generate output (stereo: left=test, right=test)
        send_data = self.send_buffer.read(self.total_samples % BUFFER_SEND_LEN, frames)
        outdata[:, 0] = send_data[:, 0]
        outdata[:, 1] = send_data[:, 0]

        # Accumulate input (left=measurement, right=reference)
        with self.lock:
            # Left channel (measurement)
            self.recv_left.write(indata[:, 0].reshape(-1, 1))
            # Right channel (reference)
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
    """

    spectrum_ready = Signal(dict)  # Emits {'magnitude': array, 'phase': array, 'frequencies': array}
    error_occurred = Signal(str)

    def __init__(self, stream_manager: AudioStreamManager):
        super().__init__()
        self.stream_manager = stream_manager
        self.running = False
        self.fft_size = stream_manager.fft_size
        self.sample_rate = stream_manager.sample_rate
        self.sparse_bins = compute_sparse_frequencies()
        self.calibration = CalibrationCoefficients()
        self.last_recv_pos = 0
        self.fft_count = 0

    def process_fft_block(self):
        """
        If a complete FFT_SIZE block is available, process it.
        """
        with self.stream_manager.lock:
            recv_write_pos = self.stream_manager.recv_write_pos
            available = (recv_write_pos - self.last_recv_pos) % BUFFER_RECV_LEN

        if available < self.fft_size:
            return  # Not enough data

        # Read FFT_SIZE samples from each channel
        left_data = self.stream_manager.recv_left.read(self.last_recv_pos, self.fft_size)
        right_data = self.stream_manager.recv_right.read(self.last_recv_pos, self.fft_size)

        self.last_recv_pos = (self.last_recv_pos + self.fft_size) % BUFFER_RECV_LEN

        # Compute FFTs
        left_spectrum = np.fft.fft(left_data[:, 0])
        right_spectrum = np.fft.fft(right_data[:, 0])

        # Compute complex quotients V at sparse frequencies
        magnitude_db = []
        phase_deg = []
        frequencies = []

        for bin_idx in self.sparse_bins:
            try:
                freq = bin_idx * (self.sample_rate / self.fft_size)
                frequencies.append(freq)

                if np.abs(right_spectrum[bin_idx]) < 1e-9:
                    # Reference too small; skip this frequency
                    magnitude_db.append(-120)
                    phase_deg.append(0)
                else:
                    V = left_spectrum[bin_idx] / right_spectrum[bin_idx]

                    # Compute impedance (simplified: just return V for now)
                    Z = V  # In v0.1, skip two-port calibration

                    magnitude_db.append(20 * np.log10(np.abs(Z) + 1e-9))
                    phase_deg.append(np.degrees(np.angle(Z)))
            except Exception as e:
                self.error_occurred.emit(f"FFT processing error: {e}")
                return

        self.fft_count += 1

        # Emit signal with results
        self.spectrum_ready.emit({
            'magnitude_db': np.array(magnitude_db),
            'phase_deg': np.array(phase_deg),
            'frequencies': np.array(frequencies),
            'fft_count': self.fft_count,
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
        
        # Create main axis for magnitude
        self.ax_mag = self.fig.add_subplot(111)
        self.ax_mag.set_xlabel('Frequency (Hz)', fontsize=10)
        self.ax_mag.set_ylabel('Magnitude (dB)', color='blue', fontsize=10)
        self.ax_mag.tick_params(axis='y', labelcolor='blue')
        self.ax_mag.set_xscale('log')
        self.ax_mag.grid(True, alpha=0.3, which='both')
        
        # Create secondary axis for phase (shares x-axis, independent y-axis)
        self.ax_phase = self.ax_mag.twinx()
        self.ax_phase.set_ylabel('Phase (°)', color='red', fontsize=10)
        self.ax_phase.tick_params(axis='y', labelcolor='red')
        self.ax_phase.set_ylim(-180, 180)  # Fixed range
        
        # Plot lines (will be created on first data)
        self.line_mag, = self.ax_mag.plot([], [], 'b-', linewidth=2, label='Magnitude', zorder=2)
        self.line_phase, = self.ax_phase.plot([], [], 'r-', linewidth=2, label='Phase', zorder=1)
        
        # Add legends
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
        
        # Auto-scale magnitude axis
        self.ax_mag.relim()
        self.ax_mag.autoscale_view(scalex=True, scaley=True)
        
        # Redraw
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

        self.setWindowTitle("Impedance Spectrum Analyzer v0.4 (matplotlib)")
        self.setGeometry(100, 100, 1600, 900)

        # Central widget layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left: Plot area (takes most space)
        self.canvas = SpectrumCanvas(self)
        main_layout.addWidget(self.canvas, 3)

        # Right: Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        # FFT_size
        control_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_spinbox = QSpinBox()
        self.fft_size_spinbox.setMinimum(1024)
        self.fft_size_spinbox.setMaximum(262144)
        self.fft_size_spinbox.setValue(FFT_SIZE)
        self.fft_size_spinbox.setEnabled(False)  # Fixed for now
        control_layout.addWidget(self.fft_size_spinbox)

        # Frequency points
        control_layout.addWidget(QLabel("Frequency Points:"))
        self.freq_points_label = QLabel(f"{len(self.dsp_worker.sparse_bins)}")
        control_layout.addWidget(self.freq_points_label)

        # Amplitude
        control_layout.addWidget(QLabel("Test Amplitude (V):"))
        self.amplitude_spinbox = QDoubleSpinBox()
        self.amplitude_spinbox.setMinimum(0.01)
        self.amplitude_spinbox.setMaximum(10.0)
        self.amplitude_spinbox.setValue(TEST_SIGNAL_AMPLITUDE / DAC_SPAN_PEAK)
        control_layout.addWidget(self.amplitude_spinbox)

        # Phase offset (display only)
        control_layout.addWidget(QLabel("Phase offset: 0°"))

        # Buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.on_start)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # Calibration buttons (placeholders)
        control_layout.addWidget(QLabel("Calibration:"))
        self.cal_open_button = QPushButton("Calibrate Open")
        self.cal_open_button.clicked.connect(self.on_calibrate_open)
        control_layout.addWidget(self.cal_open_button)

        self.cal_short_button = QPushButton("Calibrate Short")
        self.cal_short_button.clicked.connect(self.on_calibrate_short)
        control_layout.addWidget(self.cal_short_button)

        self.cal_load_button = QPushButton("Calibrate Load (100Ω)")
        self.cal_load_button.clicked.connect(self.on_calibrate_load)
        control_layout.addWidget(self.cal_load_button)

        # Status area
        control_layout.addStretch()
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        main_layout.addWidget(control_panel, 1)

        # Connect DSP signals
        self.dsp_worker.spectrum_ready.connect(self.on_spectrum_ready)
        self.dsp_worker.error_occurred.connect(self.on_dsp_error)

        # Timer for DSP processing loop
        self.dsp_timer = QTimer()
        self.dsp_timer.timeout.connect(self.dsp_worker.run_loop)
        self.dsp_timer.setInterval(50)  # Process every 50 ms

    @Slot(dict)
    def on_spectrum_ready(self, data: dict):
        """Update plot when FFT results are ready."""
        frequencies = data['frequencies']
        magnitude_db = data['magnitude_db']
        phase_deg = data['phase_deg']

        # Update matplotlib canvas
        self.canvas.update_plot(frequencies, magnitude_db, phase_deg)

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

    def on_calibrate_open(self):
        """Placeholder for open calibration."""
        self.status_label.setText("Open calibration: placeholder (v0.2+)")

    def on_calibrate_short(self):
        """Placeholder for short calibration."""
        self.status_label.setText("Short calibration: placeholder (v0.2+)")

    def on_calibrate_load(self):
        """Placeholder for load calibration."""
        self.status_label.setText("Load calibration (100Ω): placeholder (v0.2+)")

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

    # Initialize Qt application
    app = QApplication(sys.argv)

    # Select audio device
    device_id = args.device if args.device is not None else sd.default.device

    try:
        print(f"Using device {device_id}: {sd.query_devices(device_id)['name']}")
    except Exception as e:
        print(f"Error querying device: {e}", file=sys.stderr)
        sys.exit(1)

    # Create audio stream manager
    try:
        stream_manager = AudioStreamManager(device_id, SAMPLE_RATE, FFT_SIZE)
        stream_manager.start()
    except Exception as e:
        print(f"Failed to initialize audio: {e}", file=sys.stderr)
        sys.exit(1)

    # Create DSP worker
    dsp_worker = DSPWorker(stream_manager)

    # Create and show UI
    ui = ImpedanceAnalyzerUI(stream_manager, dsp_worker)
    ui.show()

    # Run Qt application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
