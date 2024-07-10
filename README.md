# FESTIM-SurfaceKinetics-Validation

## Overview

This repository contains scripts for V&V of surface kinetics model implemented in FESTIM.

## Cases

The repository includes several validation cases and one verification test on the MMS basis.

[1] H absorption in Ti: [Hamamoto, Y., Uchikoshi, T., & Tanabe, K. (2020). Comprehensive modeling of hydrogen transport and accumulation in titanium and zirconium. Nuclear Materials and Energy, 23, 100751](https://www.sciencedirect.com/science/article/pii/S2352179120300272)

[2] D adsorption on oxidised W: [Hodille, E. A., Pavec, B., Denis, J., Dunand, A., Ferro, Y., Minissale, M., ... & Bisson, R. (2024). Deuterium uptake, desorption and sputtering from W (110) surface covered with oxygen. Nuclear Fusion, 64(4), 046022](https://iopscience.iop.org/article/10.1088/1741-4326/ad2a29/pdf)

[3] D implantation in damaged Eurofer: [Schmid, K., Schwarz-Selinger, T., & Theodorou, A. (2023). Modeling intrinsic-and displacement-damage-driven retention in EUROFER. Nuclear Materials and Energy, 36, 101494](https://www.sciencedirect.com/science/article/pii/S2352179123001333)

[4] D implantation in damaged W: [Hodille, E. A., Markelj, S., Schwarz-Selinger, T., Založnik, A., Pečovnik, M., Kelemen, M., & Grisolia, C. (2018). Stabilization of defects by the presence of hydrogen in tungsten: simultaneous W-ion damaging and D-atom exposure. Nuclear Fusion, 59(1), 016011](https://iopscience.iop.org/article/10.1088/1741-4326/aaec97)

[5] MMS test

## How to use

Clone this repository to your local machine.

    ```bash
    git clone https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation
    ```

Create and activate the correct conda environment with the required dependencies:

    ```bash
    conda env create -f environment.yml
    conda activate festim-surface-kinetics-vv-env
    ```

This will set up a Conda environment named `festim-surface-kinetics-vv-env` with all the required dependencies for running the FESTIM scripts.

Navigate to the desired case folder and execute the Python scripts using the activated Conda environment.