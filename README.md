# FESTIM-SurfaceKinetics-Validation

## Overview

This repository contains scripts for V&V of surface kinetics model implemented in FESTIM.

## Cases

The repository includes several validation cases and one verification test on the MMS basis.

[1] [H absorption in Ti](https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation/tree/main/H_Ti): The case reproduces simulation results of [Y. Hamamoto et al.](https://www.sciencedirect.com/science/article/pii/S2352179120300272) on the H absorption in Ti. The simulations are based on experiments performed by [Y. Harooka et al.](https://www.sciencedirect.com/science/article/abs/pii/0022311581905663?via%3Dihub)

[2] [D adsorption on oxidised W](https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation/tree/main/D_WO): The case reproduces simulation results of [E. A. Hodille et al.](https://iopscience.iop.org/article/10.1088/1741-4326/ad2a29) on the D adsorption/desorption on/from oxidised W. The simulations are based on experiments performed by [A. Dunand et al.](https://iopscience.iop.org/article/10.1088/1741-4326/ac583a)

[3] [D implantation in damaged W](./D_damagedW): The case reproduces simulation results of [E. A. Hodille et al.](https://iopscience.iop.org/article/10.1088/1741-4326/aa5aa5/meta) The simulations are based on experiments performed by [S. Markelj et al.](https://www.sciencedirect.com/science/article/pii/S0022311515303470?via%3Dihub)

[4] MMS test

## How to use

Jupyter books in folders can be inspected online with Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KulaginVladimir/FESTIM-SurfaceKinetics-Validation/HEAD)

For a local use, clone this repository to your local machine.

```
git clone https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation
```

Create and activate the correct conda environment with the required dependencies:

```
conda env create -f environment.yml
conda activate festim-surface-kinetics-vv-env
```

This will set up a Conda environment named `festim-surface-kinetics-vv-env` with all the required dependencies for running the FESTIM scripts.

Navigate to the desired case folder and run the Jupyter books using the activated Conda environment.