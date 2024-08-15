import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import log_loss
from scipy.stats import t as student_t

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generating synthetic price movement data using Markov Chains
# States: 0 = downtrend, 1 = uptrend
states = np.array([0, 1])

# Transition matrix: probability of moving from one state to another
# Example: 0.8 probability of staying in the current state, 0.2 probability of switching
transition_matrix = np.array([[0.8, 0.2],
                              [0.3, 0.7]])

# Simulate state transitions for a certain number of time steps
n_steps = 1000
state_sequence = [np.random.choice(states, p=[0.5, 0.5])]

for _ in range(1, n_steps):
    current_state = state_sequence[-1]
    next_state = np.random.choice(states, p=transition_matrix[current_state])
    state_sequence.append(next_state)

# Simulate price movements based on the state sequence
price_movements = np.random.randn(n_steps) * (np.array(state_sequence) + 1)

# Generate cumulative price to simulate the actual price
price_series = np.cumsum(price_movements)

# Plot the price series
plt.figure(figsize=(12, 6))
plt.plot(price_series, label='Simulated Price Series')
plt.title('Simulated Price Series Using Markov Chain')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Building a Hidden Markov Model (HMM)
# HMM is used to model the observed price series and infer the hidden state sequence

# Reshape the data for the HMM model
price_series_reshaped = price_series.reshape(-1, 1)

# Define the HMM model with Gaussian emissions
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, tol=0.01)

# Fit the model to the observed price series
log_likelihood = model.fit(price_series_reshaped).score(price_series_reshaped)
print(f"Log Likelihood after training: {log_likelihood}")

# Step 3: Predicting the hidden states using the trained model
hidden_states = model.predict(price_series_reshaped)

# Visualize the hidden states with the price series
plt.figure(figsize=(12, 6))
plt.plot(price_series, label='Price Series')
plt.plot(hidden_states * max(price_series), label='Inferred Hidden States', linestyle='--', color='red')
plt.title('Hidden Markov Model: Inferred States vs. Price Series')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Analyze the log-likelihood of the model
# This evaluates how well the model explains the observed data
logprob = model.score(price_series_reshaped)
print(f"Log Likelihood of the fitted model: {logprob}")

# Step 5: Perform Baum-Welch training with multiple iterations
# The Baum-Welch algorithm is used to find the best parameters for the HMM
# We check if the log-likelihood improves with more iterations

# Re-fitting the model with more iterations and tracking convergence
model.n_iter = 200
log_likelihoods = []
for i in range(1, 6):
    model.fit(price_series_reshaped)
    log_likelihood = model.score(price_series_reshaped)
    log_likelihoods.append(log_likelihood)
    print(f"Iteration {i}, Log Likelihood: {log_likelihood}")

# Plot the log-likelihood over iterations to see convergence
plt.figure(figsize=(12, 6))
plt.plot(range(1, 6), log_likelihoods, marker='o')
plt.title('Log Likelihood Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.grid(True)
plt.show()

# Step 6: Heatmap of the original transition matrix
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=['Downtrend', 'Uptrend'],
            yticklabels=['Downtrend', 'Uptrend'])
plt.title('Original Transition Matrix Heatmap')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.show()

# Step 7: Analyzing the transition probabilities inferred by the HMM
learned_transition_matrix = model.transmat_

print("Original Transition Matrix:\n", transition_matrix)
print("Learned Transition Matrix:\n", learned_transition_matrix)

# Heatmap of the learned transition matrix
plt.figure(figsize=(8, 6))
sns.heatmap(learned_transition_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=['Downtrend', 'Uptrend'],
            yticklabels=['Downtrend', 'Uptrend'])
plt.title('Learned Transition Matrix Heatmap')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.show()

# Step 8: Predicting future price movements based on hidden states
# Use the trained HMM to predict future states and price movements
n_future_steps = 50
future_hidden_states = []

# Predict the future hidden states
last_hidden_state = hidden_states[-1]
for _ in range(n_future_steps):
    future_hidden_state = np.random.choice(states, p=model.transmat_[last_hidden_state])
    future_hidden_states.append(future_hidden_state)
    last_hidden_state = future_hidden_state

# Simulate future price movements based on the predicted hidden states
future_price_movements = np.random.randn(n_future_steps) * (np.array(future_hidden_states) + 1)
future_price_series = np.cumsum(future_price_movements) + price_series[-1]

# Plot the future price predictions
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_steps), price_series, label='Historical Price Series')
plt.plot(np.arange(n_steps, n_steps + n_future_steps), future_price_series, label='Predicted Future Price Series', linestyle='--')
plt.title('Predicted Future Price Series Using HMM')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Instead of Gaussian emissions, consider using other distributions like Student's t-distribution or mixtures of Gaussians to capture more complex patterns in price movements.

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generating synthetic price movement data using Markov Chains
states = np.array([0, 1])
transition_matrix = np.array([[0.8, 0.2],
                              [0.3, 0.7]])
n_steps = 1000
state_sequence = [np.random.choice(states, p=[0.5, 0.5])]
for _ in range(1, n_steps):
    current_state = state_sequence[-1]
    next_state = np.random.choice(states, p=transition_matrix[current_state])
    state_sequence.append(next_state)

price_movements = np.random.randn(n_steps) * (np.array(state_sequence) + 1)
price_series = np.cumsum(price_movements)

# Plot the price series
plt.figure(figsize=(12, 6))
plt.plot(price_series, label='Simulated Price Series')
plt.title('Simulated Price Series Using Markov Chain')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Building a Hidden Markov Model (HMM)
price_series_reshaped = price_series.reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
price_series_scaled = scaler.fit_transform(price_series_reshaped)

# Define HMM with Gaussian emissions
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, tol=0.01)

# Fit the model to the observed price series
try:
    model.fit(price_series_scaled)
    log_likelihood = model.score(price_series_scaled)
    print(f"Log Likelihood after training: {log_likelihood}")
except Warning as e:
    print(f"Convergence warning: {e}")

# Step 3: Predicting the hidden states using the trained model
hidden_states = model.predict(price_series_scaled)

# Visualize the hidden states with the price series
plt.figure(figsize=(12, 6))
plt.plot(price_series, label='Price Series')
plt.plot(hidden_states * max(price_series), label='Inferred Hidden States', linestyle='--', color='red')
plt.title('Hidden Markov Model: Inferred States vs. Price Series')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Analyze the log-likelihood of the model
logprob = model.score(price_series_scaled)
print(f"Log Likelihood of the fitted model: {logprob}")

# Step 5: Perform Grid Search for Hyperparameter Tuning
param_grid = {
    'n_components': [2, 3, 4],
    'covariance_type': ['diag', 'spherical', 'tied', 'full']
}

# Create a custom scorer for GridSearchCV
def log_loss_scorer(estimator, X):
    try:
        return -estimator.score(X)
    except:
        return np.nan

grid_search = GridSearchCV(estimator=hmm.GaussianHMM(), param_grid=param_grid, scoring=log_loss_scorer, cv=3)
grid_search.fit(price_series_scaled)
print("Best Parameters:", grid_search.best_params_)

# Step 6: Use Student's t-distribution for emissions
class StudentTEmissionHMM(hmm.GaussianHMM):
    def _do_e_step(self):
        super()._do_e_step()
        self.means_ = np.mean(price_series_scaled, axis=0)
        self.covars_ = np.cov(price_series_scaled, rowvar=False)
        self.covars_ = student_t(df=5).rvs(size=self.covars_.shape)

# Define the HMM with Student's t-distribution emissions
model_t = StudentTEmissionHMM(n_components=2, covariance_type="diag", n_iter=100, tol=0.01)
model_t.fit(price_series_scaled)
log_likelihood_t = model_t.score(price_series_scaled)
print(f"Log Likelihood with Student's t-distribution: {log_likelihood_t}")

# Step 7: Heatmap of the original transition matrix
plt.figure(figsize=(8, 6))
sns.heatmap(transition_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=['Downtrend', 'Uptrend'],
            yticklabels=['Downtrend', 'Uptrend'])
plt.title('Original Transition Matrix Heatmap')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.show()

# Step 8: Analyzing the transition probabilities inferred by the HMM
learned_transition_matrix = model.transmat_
print("Original Transition Matrix:\n", transition_matrix)
print("Learned Transition Matrix:\n", learned_transition_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(learned_transition_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=['Downtrend', 'Uptrend'],
            yticklabels=['Downtrend', 'Uptrend'])
plt.title('Learned Transition Matrix Heatmap')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.show()

# Step 9: Predicting future price movements based on hidden states
n_future_steps = 50
future_hidden_states = []
last_hidden_state = hidden_states[-1]
for _ in range(n_future_steps):
    future_hidden_state = np.random.choice(states, p=model.transmat_[last_hidden_state])
    future_hidden_states.append(future_hidden_state)
    last_hidden_state = future_hidden_state

future_price_movements = np.random.randn(n_future_steps) * (np.array(future_hidden_states) + 1)
future_price_series = np.cumsum(future_price_movements) + price_series[-1]

# Plot the future price predictions
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_steps), price_series, label='Historical Price Series')
plt.plot(np.arange(n_steps, n_steps + n_future_steps), future_price_series, label='Predicted Future Price Series', linestyle='--')
plt.title('Predicted Future Price Series Using HMM')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
