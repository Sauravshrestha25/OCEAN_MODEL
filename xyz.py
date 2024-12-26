import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('OCEAN MODEL.csv', sep ='\t')
print(dataset.head())
print(dataset.info())
print(dataset.describe())
data = dataset.copy()
pd.options.display.max_columns = 150
data.drop(data.columns[50:107], axis = 1, inplace = True)
data.drop(data.columns[51:], axis = 1, inplace = True)
data.head(20)

print(dataset.isnull().sum())
pd.options.display.max_columns = 150
dataset.head(5)
dataset = dataset.iloc[:, :50]

dataset.head(3)

print(dataset.isnull().sum())

dataset = dataset.dropna()
print(dataset.shape)

# Groups and Questions
ext_questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention',
                 'EXT10': 'I am quiet around strangers'}

est_questions = {'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue'}

agr_questions = {'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease'}

csn_questions = {'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule',
                 'CSN10' : 'I am exacting in my work'}

opn_questions = {'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}

# Group Names and Columns
EXT = [column for column in data if column.startswith('EXT')]
EST = [column for column in data if column.startswith('EST')]
AGR = [column for column in data if column.startswith('AGR')]
CSN = [column for column in data if column.startswith('CSN')]
OPN = [column for column in data if column.startswith('OPN')]

def vis_questions(groupname, questions, color):
    plt.figure(figsize=(40,60))
    for i in range(1, 11):
        plt.subplot(10,5,i)
        plt.hist(data[groupname[i-1]], bins=14, color= color, alpha=.5)
        plt.title(questions[groupname[i-1]], fontsize=18)

dataset['Openness'] = dataset.iloc[:, 40:50].mean(axis = 1) #OPN1 to OPN10
dataset['Conscientiousness'] = dataset.iloc[:, 30:40].mean(axis=1)  # CSN1 to CSN10
dataset['Extraversion'] = dataset.iloc[:, 0:10].mean(axis=1)  # EXT1 to EXT10
dataset['Agreeableness'] = dataset.iloc[:, 20:30].mean(axis=1)  # AGR1 to AGR10
dataset['Neuroticism'] = dataset.iloc[:, 10:20].mean(axis=1)  # EST1 to EST10

X = dataset.iloc[:, :50]  # Input features: 50 questionnaire responses
y = dataset[['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']]  # Target labels


from sklearn.model_selection import train_test_split

# Splitting for  training (80%) and testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # For progress bar

# For progress bar
print("Initializing the model...")
progress_bar = tqdm(total=3, desc="Training and Evaluation Progress", position=0)

# Creating the model (Random Forest Regressor)
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state=42, verbose =1))

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Training the model")

# Training the model
rf_model.fit(X_train , y_train)

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Making predictions on the test set")

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Evaluating the model")

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
progress_bar.close()
print("Mean Squared Error:", mse)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluating using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Evaluating using R² Score
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # For progress bar

# For progress bar
print("Initializing the model...")
progress_bar = tqdm(total=3, desc="Training and Evaluation Progress", position=0)

# Creating the model (Gradient Boosting Regressor)
gb_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Training the model")

# Training the model
gb_model.fit(X_train, y_train)

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Making predictions on the test set")

# Making predictions on the test set
y_pred = gb_model.predict(X_test)

# Updating progress bar
progress_bar.update(1)
progress_bar.set_description("Evaluating the model")

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
progress_bar.close()
print("Mean Squared Error:", mse)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluating using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Evaluating using R² Score
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for RandomForestRegressor
param_grid = {
    'estimator__n_estimators': [100, 200, 300],  # Number of trees
    'estimator__max_depth': [10, 20, 30],  # Maximum depth of trees
    'estimator__min_samples_split': [2, 5, 10]  # Minimum samples to split a node
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=MultiOutputRegressor(RandomForestRegressor(random_state=42)),
                           param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Negative MSE Score:", grid_search.best_score_)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

# Define the parameter grid for RandomForestRegressor
param_grid = {
    'estimator__n_estimators': [100, 200, 300],  # Number of trees
    'estimator__max_depth': [10, 20, 30],  # Maximum depth of trees
    'estimator__min_samples_split': [2, 5, 10]  # Minimum samples to split a node
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(estimator=MultiOutputRegressor(RandomForestRegressor(random_state=42)),
                           param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)

# Create a custom progress bar based on the number of parameter combinations
total_combinations = len(param_grid['estimator__n_estimators']) * len(param_grid['estimator__max_depth']) * len(param_grid['estimator__min_samples_split'])
progress_bar = tqdm(total=total_combinations, desc="GridSearchCV Progress", position=0)

# Custom callback to update the progress bar
def update_progress(*args, **kwargs):
    progress_bar.update(1)

# Attach the callback function to the grid search
grid_search.fit(X_train, y_train)

# Close the progress bar
progress_bar.close()

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Negative MSE Score:", grid_search.best_score_)

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Reduce dataset size if needed
X_sample = X_train.sample(10000, random_state=42)  # Optional
y_sample = y_train.loc[X_sample.index]

# Parameter grid
param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [10, 20],
    'estimator__min_samples_split': [2, 5]
}

# HalvingGridSearchCV
halving_search = HalvingGridSearchCV(
    estimator=MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1)),
    param_grid=param_grid, cv=2, scoring='neg_mean_squared_error', verbose=1)

halving_search.fit(X_sample, y_sample)
print("Best Parameters:", halving_search.best_params_)
print("Best Negative MSE Score:", halving_search.best_score_)

final_model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1))

final_model.fit(X_train, y_train)
print("Final model trained with best parameters.")

y_pred = final_model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Create a DataFrame for the input
columns = X_train.columns  # Use the same feature names as in the training data
sample_user_input_df = pd.DataFrame(sample_user_input, columns=columns)

# Make predictions
predicted_traits = final_model.predict(sample_user_input_df)
predicted_traits = predicted_traits.flatten()  # Flatten the output if it's in 2D
print("Predicted Personality Traits (OCEAN):", predicted_traits)

# Define traits and values
traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
predicted_values = predicted_traits  # Use the predicted values directly

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(traits, predicted_values, color=['skyblue', 'orange', 'green', 'purple', 'red'])
plt.ylim(0, 5)  # Assuming the range is 1 to 5
plt.title("Predicted OCEAN Personality Traits", fontsize=16)
plt.xlabel("Traits", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(np.arange(0, 6, 1), fontsize=12)
plt.tight_layout()
plt.show()

# Radar Chart
# Prepare data for radar chart
values = list(predicted_values) + [predicted_values[0]]  # Close the circle
angles = [n / float(len(traits)) * 2 * pi for n in range(len(traits))]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plot data
ax.plot(angles, values, linewidth=2, linestyle='solid', label="Predicted Traits")
ax.fill(angles, values, alpha=0.3)

# Add labels and title
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], color="grey", size=12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(traits, fontsize=14)

plt.title("OCEAN Personality Traits Radar Chart", size=16, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Make predictions on the test set
y_pred_test = final_model.predict(X_test)

# Convert predictions and true values to 1D arrays for comparison (if multi-output)
y_pred_test_flat = np.array(y_pred_test).flatten()
y_test_flat = np.array(y_test).flatten()

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_flat, y=y_pred_test_flat, alpha=0.7, color='blue')
plt.plot([0, 5], [0, 5], '--r', label='Perfect Prediction')  # Line y=x for reference
plt.title("Actual vs Predicted Personality Traits", fontsize=16)
plt.xlabel("Actual Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Residual plot
residuals = y_test_flat - y_pred_test_flat
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green', bins=20, alpha=0.7)
plt.title("Residual Distribution (Actual - Predicted)", fontsize=16)
plt.xlabel("Residuals", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

import joblib

# Save the trained model to a file
joblib.dump(final_model, 'final_personality_model.pkl')

print("Model saved successfully!")
