# MLniCS

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

MLniCS is a library for building physics-informed neural networks (PINNs), physics-reinforced neural networks (PRNNs), and projection-driven neural networks (PDNNs) in the context of parameterized reduced order models. For more about these methodologies, see the paper [Physics-informed machine learning for reduced-order modeling of nonlinear problems](https://www.sciencedirect.com/science/article/pii/S0021999121005611).

## Requirements

Before using MLniCS, it is necessary to install:
* FEniCS (>= 2018.1.0)
* [RBniCS](https://www.rbnicsproject.org/)

## Summary of MLniCS Features

MLniCS provides users functions for
* Initializing training, validation, and testing sets
* PINN, PRNN, and PDNN losses
* Training PINNs, PRNNs, and PDNNs
* Computing errors for PINNs, PRNNs, and PDNNs
* Saving neural network weights and training parameters

In this repository, there are also tutorials with sample problems using MLniCS.
