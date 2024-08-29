# Financial Models Library

This repository provides Python implementations of key financial models commonly used in quantitative finance and high-frequency trading. The models included are:

- **Black-Scholes Model**: A classical model used for pricing European-style options.
- **Heston Model**: A stochastic volatility model that extends the Black-Scholes framework to account for changing volatility.
- **GARCH Model**: A model for estimating time-varying volatility based on historical data.
- **Kalman Filter Model**: 

## Overview

This repository is designed for practise and documentation of my learning processes (learning About trading algorithms).

## Models

### [Black-Scholes Model](https://github.com/CodingLife1024/High-Frequency-Trading/tree/master/Black%20Scholes%20Model)

The Black-Scholes model is used to calculate the theoretical price of European call and put options based on assumptions about constant volatility and interest rates.

**Key Features:**
- Option pricing formulas for call and put options.
- Calculation of Greeks (Delta, Gamma, Vega, Theta, Rho).

### Heston Model

The Heston model incorporates stochastic volatility, allowing for more realistic modeling of market conditions compared to the Black-Scholes model.

**Key Features:**
- Option pricing using the Heston closed-form solution.
- Calibration of model parameters using historical data.
- Simulation of paths for underlying asset prices and volatilities.

### GARCH Model

The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model estimates time-varying volatility based on past returns.

**Key Features:**
- Estimation of volatility using GARCH(1,1) and other variants.
- Implementation of parameter estimation and forecasting.
- Visualization of volatility clusters and volatility forecasts.

## Installation

To use the models, clone the repository and install the required dependencies.

```bash
git https://github.com/CodingLife1024/High-Frequency-Trading
cd High-Frequency-Trading
pip install -r requirements.txt
```
