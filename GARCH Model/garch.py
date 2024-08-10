import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model

# Simulate some data
np.random.seed(42)
n = 1000
omega = 0.2
alpha = 0.5
beta = 0.4

# Generate GARCH(1,1) process
returns = np.random.normal(size=n)
volatility = np.zeros(n)
volatility[0] = np.std(returns)
for t in range(1, n):
    volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
    returns[t] *= volatility[t]

# Convert to a DataFrame
data = pd.DataFrame({'Returns': returns})

# Fit a GARCH(1,1) model
model = arch_model(data['Returns'], vol='Garch', p=1, q=1)
garch_fit = model.fit()

# Print the summary
print(garch_fit.summary())

# Forecast volatility
forecasts = garch_fit.forecast(horizon=10)
forecasted_volatility = forecasts.variance[-1:]

# Plot the simulated returns
plt.figure(figsize=(10, 6))
plt.plot(data['Returns'], label='Returns')
plt.title('Simulated Returns')
plt.legend()
plt.show()

# Plot the fitted conditional volatility
plt.figure(figsize=(10, 6))
plt.plot(garch_fit.conditional_volatility, label='Conditional Volatility', color='orange')
plt.title('Fitted Conditional Volatility')
plt.legend()
plt.show()

# Rolling window volatility plot
rolling_window = 100
rolling_volatility = data['Returns'].rolling(window=rolling_window).std()

plt.figure(figsize=(10, 6))
plt.plot(rolling_volatility, label=f'Rolling Window Volatility ({rolling_window} periods)', color='green')
plt.title('Rolling Window Volatility')
plt.legend()
plt.show()

# Heatmap of conditional variances
cond_vars = garch_fit.conditional_volatility ** 2
data['Conditional Variance'] = cond_vars

# Create heatmap data
heatmap_data = pd.DataFrame({
    'Conditional Variance': cond_vars
})

# Reshape the data for the heatmap
heatmap_matrix = heatmap_data.values.reshape(-1, 50)  # Reshape to 50 columns

# Normalize for better heatmap visibility
heatmap_matrix_normalized = (heatmap_matrix - heatmap_matrix.min()) / (heatmap_matrix.max() - heatmap_matrix.min())

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_matrix_normalized, cmap='YlGnBu', annot=False, linewidths=0.5)
plt.title('Heatmap of Conditional Variances')
plt.xlabel('Time')
plt.ylabel('Conditional Variance')
plt.show()

# Print forecasted volatility
print(f"Forecasted Volatility for the next 10 periods: {forecasted_volatility.values}")