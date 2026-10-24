# Impedance Spectrometer V06

This directory is the active continuation of the working impedance analyzer baseline.
It sits beside the older versioned folders in the repository root instead of being nested under a second `programming_experiments` directory level.

Current status:
- working sparse FFT / multi-tone excitation logic retained from V05
- progress-bar text/value behavior preserved and ready for display
- soundcard filtering and menu scaffolding added for the next review pass
- JSON config/calibration save/load is still planned for later

Key changes in this version:
- list only soundcards with at least 2 input and 2 output channels
- prepare runtime soundcard/sample-rate switching by reusing a restart path in the stream manager
- add top-level menu structure: File, Soundcard, About
- About dialog shows a short V0.6 banner
- File menu actions are placeholders until the configuration/calibration save format is implemented
