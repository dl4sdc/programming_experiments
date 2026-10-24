# Impedance Spectrometer V06

This directory is a clean continuation of the working proof-of-concept from
`impedance_spectrometer_V01/impedance_meas_V05.py`.

Purpose:
- preserve the known-good sparse FFT multi-tone excitation + readback flow
- continue development in a fresh directory instead of patching the older file set
- add missing features incrementally without losing the stable baseline

Planned work:
- verify signal generation and FFT analysis against calibration data
- improve UI and plotting behavior
- add save/load calibration, better progress display, and stability fixes
- keep the core implementation aligned with PySide6 / sounddevice / numpy / matplotlib versions in this environment

The main script here is `impedance_meas_V06.py` and is intentionally seeded from the last working code before further edits.
