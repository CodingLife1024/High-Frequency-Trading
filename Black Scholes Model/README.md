The Black-Scholes model is a mathematical framework and algorithm for pricing European-style options, which are financial derivatives that give the holder the right, but not the obligation, to buy or sell an underlying asset at a specified price on a specified date.

### Introduction

Here are some basic definitions of some of the terms used in this model - 

__Strike Price (K)__: The strike price, also known as the exercise price, is the price at which the option holder can buy or sell the underlying asset. In the context of a call option, the strike price is the price at which the holder can purchase the asset. For a put option, it is the price at which the holder can sell the asset.

__Underlying Asset Price/Stock Price (S)__: The current price of the asset on which the option is written. This is the price that will be compared to the strike price to determine the payoff of the option at expiration.

__Volatility ($\sigma$)__: Volatility refers to the degree of variation in the price of the underlying asset over time. In the Black-Scholes model, volatility is a key input and is usually expressed as the standard deviation of the asset's returns. It reflects the market's expectation of the asset's future price fluctuations.

Mathematical Expression:

$$
\sigma = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (r_t - \bar{r})^2}
$$

__Time to Maturity (in years) (T)__: This represents the remaining time until the option's expiration date.

__Risk-Free Interest Rate (r)__: The risk-free interest rate is the theoretical rate of return on an investment with zero risk of financial loss. It is often represented by the yield on government bonds. In the Black-Scholes model, the risk-free rate is used to discount the expected payoff of the option to present value.

__Option Price (C or P)__: The option price refers to the premium or cost of the option. The Black-Scholes model provides a formula to calculate the theoretical price of both call options (denoted as C) and put options (denoted as P).

Mathematical Expressions: 

Call Option Price (C):

$
C = S_0 \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
$

Put Option Price (P):

$
P = K \cdot e^{-rT} \cdot N(-d_2) - S_0 \cdot N(-d_1)
$

Definitions of d1 and d2:

$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}
$

$
d_2 = d_1 - \sigma\sqrt{T}
$

where, 

$
N(d) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{d} e^{-\frac{x^2}{2}} \, dx
$

N(d) is the cumulative distribution function of the standard normal distribution.

__The Greeks__: The "Greeks" are various risk measures that describe how the option price is sensitive to various factors like the underlying asset's price, volatility, and time. Some common Greeks include:

1. __Delta ($\Delta$)__: Measures the sensitivity of the option price to changes in the underlying asset's price.

2. __Gamma ($\Gamma$)__: Measures the sensitivity of Delta to changes in the underlying asset's price.

3. __Theta ($\Theta$)__: Measures the sensitivity of the option price to the passage of time (time decay).

4. __Vega ($\nu$)__: Measures the sensitivity of the option price to changes in volatility.

5. __Rho ($\rho$)__: Measures the sensitivity of the option price to changes in the risk-free interest rate.

### Derivation of the Black-Scholes Equation

The Black-Scholes equation is a partial differential equation that describes the price of the option as a function of the underlying asset price and time. The derivation of this equation is based on constructing a risk-free portfolio and applying Ito's Lemma, which is a key result from stochastic calculus.

#### Step 1: Model Assumptions

1. The underlying asset price \( $S_t$ \) follows a geometric Brownian motion:

   $
   dS_t = \mu S_t dt + \sigma S_t dW_t
   $

   where:

   - \( $\mu$ \) is the drift rate (expected return) of the asset.

   - \( $\sigma$ \) is the volatility of the asset.

   - \( $W_t$ \) is a Wiener process (standard Brownian motion).

2. The risk-free rate \( r \) is constant over time.

3. No arbitrage opportunities exist.

#### Step 2: Portfolio Construction

Consider a portfolio \( $\Pi$ \) consisting of a long position in one option and a short position in \( $\Delta$ \) units of the underlying asset. The value of the portfolio at time \( t \) is:

$\Pi$ = V($S_t$, t) - $\Delta$ $S_t$

where \( V($S_t$, t) \) is the value of the option at time \( t \).

#### Step 3: Applying Ito's Lemma

Using [Ito's Lemma](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma), we can express the differential of the option price \( V \) as:

$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} dS_t + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} (dS_t)^2
$

Substituting the expression for \( $dS_t$ \) from the geometric Brownian motion model, we get:

$
dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} (\mu S_t dt + \sigma S_t dW_t) + \frac{1}{2} \frac{\partial^2 V}{\partial S^2} \sigma^2 S_t^2 dt
$

Simplifying, we obtain:

$
dV = \left(\frac{\partial V}{\partial t} + \mu S_t \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2}\right) dt + \sigma S_t \frac{\partial V}{\partial S} dW_t
$

#### Step 4: Eliminating Risk

For the portfolio \( $\Pi$ \) to be risk-free, the term involving \( $dW_t$ \) must be zero. Therefore, we set:

$
\Delta = \frac{\partial V}{\partial S}
$

Thus, the portfolio value becomes:

$
\Pi = V(S_t, t) - \frac{\partial V}{\partial S} S_t
$

The change in the portfolio value over a small time interval \( dt \) is:

$
d\Pi = dV - \Delta dS_t
$

Substituting the expressions for \( dV \) and \( $dS_t$ \), we get:

$
d\Pi = \left(\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2}\right) dt
$

#### Step 5: No-Arbitrage Condition

Since the portfolio is risk-free, it must earn the risk-free rate \( r \). Therefore:

$
d\Pi = r\Pi dt
$

Substituting 

$
\Pi = V - S_t \frac{\partial V}{\partial S}
$

into the equation above:

$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} + r S_t \frac{\partial V}{\partial S} - r V = 0
$

#### Step 6: The Black-Scholes Equation
Rearranging the terms, we obtain the Black-Scholes partial differential equation:

$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S_t^2 \frac{\partial^2 V}{\partial S^2} + r S_t \frac{\partial V}{\partial S} - r V = 0
$

This equation is the foundation for pricing European options.

#### Boundary Conditions

For a European call option with strike price \( K \) and expiration time \( T \), the boundary condition is given by:

$
V(S_T, T) = \max(S_T - K, 0)
$

For a European put option, the boundary condition is:

$
V(S_T, T) = \max(K - S_T, 0)
$

The Black-Scholes equation, along with the boundary conditions, can be solved to find the price of European options.

### Derivation of the Black-Scholes Formula by Applying Boundary Conditions

Let's derive the Black-Scholes formula for European call and put options by applying the boundary conditions to the Black-Scholes partial differential equation (PDE).

#### Step 1: Recall the Black-Scholes PDE

The Black-Scholes PDE for the value \( V(S, t) \) of an option is given by:

$
\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV
$

where:
- \( S \) is the underlying asset price,

- \( t \) is time,

- \( r \) is the risk-free interest rate,

- \( $\sigma$ \) is the volatility of the asset.

#### Step 2: Boundary Conditions for Call and Put Options

1. **Call Option**: The boundary condition at maturity \( t = T \) for a European call option with strike price \( K \) is:

   $
   C(S_T, T) = \max(S_T - K, 0)
   $

   This means that at expiration, the value of the call option is the difference between the stock price and the strike price if the stock price is above the strike price, otherwise, it is zero.

2. **Put Option**: The boundary condition at maturity \( t = T \) for a European put option with strike price \( K \) is:

   $
   P(S_T, T) = \max(K - S_T, 0)
   $

   This means that at expiration, the value of the put option is the difference between the strike price and the stock price if the strike price is above the stock price, otherwise, it is zero.

#### Step 3: Transforming the PDE

The Black-Scholes partial differential equation (PDE) for the value \( V(S, t) \) of an option is given by:

$
\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV
$

#### 1. Change of Variables

To solve this PDE, we make a change of variables to simplify the equation:

- Let 
$
\tau = T - t
$, 

where \( $\tau$ \) represents the time to maturity. As time progresses towards maturity, \( $\tau$ \) decreases to zero.
  
- Define 
$
x = \ln(S),
$
a logarithmic transformation of the stock price.

This change of variables is useful because it simplifies the nonlinear term 
$
S \frac{\partial V}{\partial S}
$
in the PDE.

#### 2. Transformation to Simplify the PDE

We propose that the solution \( $V(S, t)$ \) can be factored into a product of a known function \( $e^{-r(T-t)}$ \) and an unknown function 
$
u(x, \tau)$:

$
V(S, t) = e^{-r(T-t)} u(x, \tau)
$

Here's the reasoning behind this transformation:

- The term 
$
e^{-r(T-t)}
$

is chosen because it matches the discounting factor for the option's future payoff under the risk-free rate \( r \). This helps to "remove" the \( rV \) term on the right-hand side of the PDE.
  
- The function 
$
u(x, \tau)
$
is an unknown function of the new variables \( x \) and \( $\tau$ \) that we want to solve for.

#### 3. Substituting into the Black-Scholes PDE

Next, we substitute 

$
V(S, t) = e^{-r(T-t)} u(x, \tau)
$

into the original Black-Scholes PDE.

First, we calculate the partial derivatives:

$
\frac{\partial V}{\partial t} = -re^{-r(T-t)} u(x, \tau) + e^{-r(T-t)} \frac{\partial u}{\partial \tau}
$

$
\frac{\partial V}{\partial S} = e^{-r(T-t)} \frac{\partial u}{\partial S} = e^{-r(T-t)} \frac{\partial u}{\partial x} \cdot \frac{1}{S}
$

$
\frac{\partial^2 V}{\partial S^2} = e^{-r(T-t)} \left(\frac{\partial^2 u}{\partial x^2} \cdot \frac{1}{S^2} - \frac{\partial u}{\partial x} \cdot \frac{1}{S^2} \right)
$

Substituting these into the Black-Scholes PDE gives:

$
\left[-re^{-r(T-t)} u + e^{-r(T-t)} \frac{\partial u}{\partial \tau} \right] + rS e^{-r(T-t)} \frac{1}{S} \frac{\partial u}{\partial x} + \frac{1}{2} \sigma^2 S^2 e^{-r(T-t)} \frac{1}{S^2} \frac{\partial^2 u}{\partial x^2} = r e^{-r(T-t)} u
$

Simplifying this, we get:

$
e^{-r(T-t)} \left[\frac{\partial u}{\partial \tau} + \frac{1}{2} \sigma^2 \frac{\partial^2 u}{\partial x^2} \right] = 0
$

This simplifies to the following heat equation for \( $u(x, \tau)$ \):

$
\frac{\partial u}{\partial \tau} = \frac{1}{2} \sigma^2 \frac{\partial^2 u}{\partial x^2}
$

#### 4. Resulting Equation

The transformation 

$
V(S, t) = e^{-r(T-t)} u(x, \tau)
$

reduces the Black-Scholes PDE to a simpler form, known as the heat equation, which is much easier to solve. The heat equation is a well-known PDE with established methods for finding solutions, which allows us to proceed with solving for 
$
u(x, \tau)
$
and thus obtaining the option price \( V(S, t) \).

#### Step 4: Solve the Heat Equation

The heat equation can be solved using standard techniques. The solution \( $u(x, \tau)$ \) will depend on the boundary conditions applied at \( \tau = 0 \) (which corresponds to \( t = T \)).

For a **call option**:

$
u(x, 0) = \max(e^x - K, 0)
$

For a **put option**:

$
u(x, 0) = \max(K - e^x, 0)
$

#### Step 5: Use the Solution to the Heat Equation

The general solution to the heat equation is a convolution of the initial condition with the heat kernel. Applying this to our transformed variables, we eventually recover the Black-Scholes formula.

#### Step 6: Applying the Solution Back to Original Variables

Transforming back to the original variables, we derive the famous Black-Scholes formulas:

- **Call Option**:

  $
  C(S_0, t) = S_0 N(d_1) - K e^{-r(T-t)} N(d_2)
  $
  
- **Put Option**:

  $
  P(S_0, t) = K e^{-r(T-t)} N(-d_2) - S_0 N(-d_1)
  $

where:

$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}
$

$
d_2 = d_1 - \sigma\sqrt{T}
$

These formulas are derived from the Black-Scholes PDE by applying the appropriate boundary conditions for European call and put options. The boundary conditions enforce that the option's value at maturity must match the payoff structure of the option, which then propagates backward through time to give the current option price.
