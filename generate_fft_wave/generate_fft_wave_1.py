#!/usr/bin/env python3
import math
import os
import sys
import wave

import numpy as np

try:
    import readline
except ImportError:
    readline = None


def safe_input(prompt: str) -> str:
    if readline is not None:
        readline.parse_and_bind("tab: complete")
        readline.set_history_length(1000)
    return input(prompt)


def prompt_int(prompt: str, min_value=None, max_value=None) -> int:
    while True:
        try:
            value = int(safe_input(prompt))
        except ValueError:
            print("Invalid integer. Please enter a whole number.", file=sys.stderr)
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.", file=sys.stderr)
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}.", file=sys.stderr)
            continue
        return value


def prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(safe_input(prompt))
        except ValueError:
            print("Invalid float. Please enter a number.", file=sys.stderr)


def prime_factors_valid(n: int) -> bool:
    """Return True if all prime factors of n are in {2,3,5,7}."""
    if n <= 0:
        return False

    x = n
    for p in (2, 3, 5, 7):
        while x % p == 0:
            x //= p

    return x == 1


def build_fft_spectrum(fft_size: int, bin_number: int, amplitude: float, phase_deg: float) -> np.ndarray:
    fft_spec = np.zeros(fft_size, dtype=np.complex128)

    theta = math.radians(phase_deg)
    pos = amplitude * complex(math.cos(theta), math.sin(theta))
    neg = amplitude * complex(math.cos(theta), -math.sin(theta))

    if fft_size == 1:
        fft_spec[0] = amplitude * complex(math.cos(theta), 0.0)
        return fft_spec

    if bin_number == 0:
        fft_spec[0] = 2.0 * amplitude * complex(math.cos(theta), 0.0)
        return fft_spec

    if fft_size % 2 == 0 and bin_number == fft_size // 2:
        fft_spec[bin_number] = 2.0 * amplitude * complex(math.cos(theta), 0.0)
        return fft_spec

    mirror = (fft_size - bin_number) % fft_size
    if mirror == bin_number:
        fft_spec[bin_number] = amplitude * complex(math.cos(theta), 0.0)
    else:
        fft_spec[bin_number] = pos
        fft_spec[mirror] = neg

    return fft_spec


def normalize_time_signal(signal: np.ndarray, target_peak: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return np.zeros_like(signal)
    return signal * (target_peak / peak)


def write_wave_file(filename: str, pcm_data: np.ndarray, sample_rate: int = 48000):
    if os.path.exists(filename):
        while True:
            ans = safe_input(f"File '{filename}' exists. Overwrite? [y/N]: ").strip().lower()
            if ans in ("", "n", "no"):
                print("Aborted without overwriting the file.", file=sys.stderr)
                raise SystemExit(0)
            if ans in ("y", "yes"):
                break
            print("Please answer yes or no.", file=sys.stderr)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())


def main():
    print("FFT waveform generator")
    print("Use the arrow keys to edit previous inputs if readline is available.")
    print()

    filename = safe_input("filename: ").strip()
    if filename == "":
        print("Filename cannot be empty.", file=sys.stderr)
        raise SystemExit(1)

    fft_size = prompt_int("FFT_size: ", min_value=1)
    if not prime_factors_valid(fft_size):
        print(f"Error: FFT_size={fft_size} has a prime factor greater than 7.", file=sys.stderr)
        raise SystemExit(1)

    bin_number = prompt_int("bin_number: ", min_value=0, max_value=fft_size - 1)
    amplitude = prompt_float("amplitude: ")
    phase_deg = prompt_float("phase_deg: ")

    fft_spec = build_fft_spectrum(fft_size, bin_number, amplitude, phase_deg)

    fft_wave = np.fft.ifft(fft_spec)
    fft_wave = np.real(fft_wave)

    fft_wave = normalize_time_signal(fft_wave, amplitude)

    normalized = fft_wave / amplitude if amplitude != 0 else np.zeros_like(fft_wave)

    if np.any(normalized < -1.0) or np.any(normalized > 1.0):
        print(
            "Warning: some samples exceed [-1.0, +1.0] after FFT normalization; clipping to full-scale PCM.",
            file=sys.stderr,
        )

    pcm_float = np.clip(normalized, -1.0, 1.0) * 32767.0
    pcm_int16 = np.rint(pcm_float).astype(np.int16)

    stereo = np.empty(pcm_int16.size * 2, dtype=np.int16)
    stereo[0::2] = pcm_int16
    stereo[1::2] = pcm_int16

    write_wave_file(filename, stereo, sample_rate=48000)

    print(f"Wave file written to '{filename}'")
    print("Done.")
    raise SystemExit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
