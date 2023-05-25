# Quantum-enhanced-variational-autoencdoer
This repository contains the source code for the paper: [Learning hard distributions with quantum-enhanced Variational Autoencoders](https://arxiv.org/abs/2305.01592)

- The sourcecode is available under the folder `Sourcecode` with sample notebooks in `Notebooks`. 
- All datasets used for the experiments are present in `Datasets`. These include 4-qubit and 8-qubit states obtained from product states, haar states, quantum-circuit states, and quantum-kicked rotor states.

## Installation

1. Clone this Github repository using the following command in your command line/terminal : <br>
```git clone git@github.com:Anantha-Rao12/Quantum-enhanced-variational-autoencoder.git```

2. Create a Python (>=3.2) virtual environemnt and call it 'Decoding-Quantum-States-with-NMR-env'.
  - On Linux/ MacOS : ```python3 -m venv qiskitml-env```
  - On Windows : ```python -m venv qiskitml-env```

A new directory called `qiskitml-env` will be created. 

3. Activate the Virtual Environment by running:
  - On Linux/ MacOS: ```source qiskitml-env/bin/activate```
  - On Windows: ```.\qiskitml-env\Scripts\activate```

4. In the new virtual environemnt , run ```pip3 install -r requirements.txt``` to install all dependencies. On Windows, `pip3` should be replaced by `pip`.

You are ready to start experiemnting with the code!

## Example notebooks

Under the `Notebooks` directory, we have two notebooks that show how we train QeVAEs for two tasks: (1) Learning the distribution (2) Compressing the circuit. The respective notebooks are titled `4qubit_productstates_analysis.ipynb` and `Circuit_compilation_QeVAE.ipynb` contain more details on implementation.

## Contact

DM Anantha S Rao - [@anantharao00](https://twitter.com/anantharao00) <br>
For clarifications and queries -- [Anantha Rao](mailto:aanantha.s.rao@gmail.com?subject=[QeVAE2023]) @2023

Project Link: [https://github.com/AnanthaRao-12/Decoding-Quantum-States-with-NMR](https://github.com/Anantha-Rao12/Quantum-enhanced-variational-autoencoder)
