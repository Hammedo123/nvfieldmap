# nvfieldmap

Open calibration and field-mapping pipeline for wide-field continuous-wave
optically detected magnetic resonance (CW-ODMR) with nitrogen-vacancy (NV)
diamond ensembles.

It turns a stack of camera frames recorded during a microwave sweep into a
magnetic-field map with a per-bin uncertainty budget:

1. camera-frame corrections (dark, bias, flat field, or off-resonance normalisation)
2. data-cube assembly, repeat averaging and pixel binning
3. multi-Lorentzian fitting with quality flags (window edge, resolvability, dip significance)
4. field inversion with the exact NV ground-state Hamiltonian
5. uncertainty budget and a synthetic-recovery validation

Status: development version, not yet released.

## Installation
Requires Python 3.10 or newer.

    git clone https://github.com/<your-username>/nvfieldmap.git
    cd nvfieldmap
    pip install -e ".[test]"

## Quick start
    pytest -q                      # run the test suite
    nvfieldmap run config.yaml     # process one run

## Physics conventions
Frequencies in MHz, fields in mT. For a field along an NV axis the two
transitions are f± = D ± γB∥, so the dip-pair splitting is 2γB∥ and

    B∥ = sqrt(((f+ − f−)/2)² − E²) / γ,   γ = 28.03 MHz/mT (g = 2.0028).

An ensemble spectrum determines |B| and the set of projections on the four NV
axes, not the lab-frame direction of B; the code reports only what is identifiable.

## Citing
See `CITATION.cff`. A DOI will be added at the first release.

## License
MIT (see `LICENSE`).
