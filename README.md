# Financial Models Library

This repository provides Python implementations of key financial models commonly used in quantitative finance and high-frequency trading. The models included are:

- **Black-Scholes Model**: A classical model used for pricing European-style options.
- **Heston Model**: A stochastic volatility model that extends the Black-Scholes framework to account for changing volatility.
- **GARCH Model**: A model for estimating time-varying volatility based on historical data.

## Overview

This repository is designed for educational purposes and to serve as a foundation for developing more complex trading strategies and risk management tools. Each model is implemented in Python and includes example usage and basic parameter calibration.

## Models

### Black-Scholes Model

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
cd financial-models
pip install -r requirements.txt
```