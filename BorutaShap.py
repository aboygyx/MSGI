# %% Data Preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sys

# Load the dataset
df_loaded = pd.read_csv('')

# Separate features (X) and the target variable (y)
X = df_loaded.iloc[:, :-1]
y = df_loaded.iloc[:, -1]

# Perform train-test split to preserve temporal or structural dependencies
# Note: Stratify is disabled as this is a regression task.
# Split ratios correspond to specific tasks (e.g., Task 1: 1867/3437 for Android, Task 4: 3044/5657 for OpenStack)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - 1867/3437, random_state=42
)

# %% Random Forest Model Initialization & Complexity Analysis
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Train the model on the training set
rf.fit(X_train, y_train)

# Compute the total number of parameters across all decision trees
total_params = 0

for tree in rf.estimators_:
    total_params += (
        len(tree.tree_.feature) +
        len(tree.tree_.threshold) +
        len(tree.tree_.children_left) +
        len(tree.tree_.children_right) +
        tree.tree_.value.size
    )

# All parameters in the Random Forest architecture are trainable
trainable_params = total_params

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}\n")

# %% Boruta-SHAP Feature Selection
_local_path = sys.path.pop(0)
from BorutaShap import BorutaShap
sys.path.insert(0, _local_path)
import matplotlib.pyplot as plt

# Initialize the Boruta-SHAP selector
Feature_Selector = BorutaShap(
    model=rf,
    importance_measure='shap',
    classification=False
)

# Execute the feature selection process to eliminate redundant variables
Feature_Selector.fit(
    X=X_train,
    y=y_train,
    n_trials=200,         # Number of iterations for statistical testing
    sample=False,         # Disable sampling to utilize the entire dataset
    train_or_test='test', # Calculate SHAP values based on the test set for unbiased evaluation
    normalize=True,       # Apply normalization to the feature space
    verbose=True
)

# Visualize the feature selection results and explicitly close the figure buffer
Feature_Selector.plot(which_features='all')
plt.close()

# Retrieve and print the selected optimal feature subset
selected_features = Feature_Selector.Subset().columns
print("Selected features by Boruta-SHAP:", list(selected_features))

# %% Export Selected Features to CSV
# Extract the retained features for both training and testing sets
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Combine the selected features with the target variable and export to CSV
train_output = X_train_selected.copy()
train_output['Target'] = y_train.values
train_output.to_csv('train_selected_features.csv', index=False)

test_output = X_test_selected.copy()
test_output['Target'] = y_test.values
test_output.to_csv('test_selected_features.csv', index=False)

print("Train and test datasets with selected features have been successfully saved to CSV.\n")

# %% SHAP Value Visualization
import shap
import numpy as np

# Configure matplotlib for academic publication (Times New Roman)
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Initialize the TreeExplainer and compute SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# Ensure the SHAP values array is strictly 2D for regression visualization
if isinstance(shap_values, list):
    shap_values = shap_values[0]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 0]

# --- 1. SHAP Summary Plot (Beeswarm) ---
plt.figure(figsize=(10, 8))

# Utilize a continuous coolwarm colormap for academic visual standards
shap.summary_plot(
    shap_values,
    X_train,
    plot_type="dot",
    color=plt.get_cmap('coolwarm'),
    show=False
)

plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14, pad=15)
plt.gca().tick_params(labelsize=11)
plt.tight_layout()

plt.savefig(
    'shap_beeswarm.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close() # Clear the canvas to prevent overlap

# --- 2. SHAP Feature Importance (Bar Plot) ---
plt.figure(figsize=(10, 8))

# Apply a low-saturation pastel color (#A8DADC) for the bar plot
shap.summary_plot(
    shap_values,
    X_train,
    plot_type="bar",
    color="#A8DADC",
    show=False
)

plt.title("SHAP Feature Importance", fontsize=14, pad=15)
plt.gca().tick_params(labelsize=11)
plt.tight_layout()

plt.savefig(
    'shap_importance.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
plt.close()
