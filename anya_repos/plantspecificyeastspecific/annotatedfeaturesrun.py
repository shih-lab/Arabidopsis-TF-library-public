import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap

dataset = pd.read_csv("/Users/anyalachwani/Downloads/June26combined_activator_dataset1_finalData1.csv")

print("Dataset shape:", dataset.shape)

label_col = dataset.columns[-1]

class0 = dataset[dataset[label_col] == 0].sample(
    n=800,
    random_state=42
)

class1 = dataset[dataset[label_col] == 1]

dataset = pd.concat([class0, class1], axis=0)

# Shuffle 
dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDataset after class-0 subsampling:")
print(dataset[label_col].value_counts().sort_index())
print("New dataset shape:", dataset.shape)

# Column P = index 15 (0-based), hence I put index 14 
sequence_col_idx = 14

# Last column is label
label_col = dataset.columns[-1]

# Features are everything after and including column P up to but not including label(the last column)
feature_cols = dataset.columns[sequence_col_idx + 1:-1]

print("\nSequence column:", dataset.columns[sequence_col_idx])
print("Label column:", label_col)
print("Number of features:", len(feature_cols))

X = dataset[feature_cols]
y = dataset[label_col]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Find columns with NaNs using training data only
nan_counts = X_train.isna().sum()
nan_cols = nan_counts[nan_counts > 0]

# Drop those columns from both train and test
X_train = X_train.drop(columns=nan_cols.index)
X_test = X_test.drop(columns=nan_cols.index)

print("Dropped", len(nan_cols), "columns.")
print("Remaining features:", X_train.shape[1])
print("\nDropped columns:")
for col in nan_cols.index:
    print(col)

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())



model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)


# Select Top 300 features using RF importance

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

top300_features = importance_df.head(100)["Feature"].tolist()

print("\nTop 10 RF Features:")
print(importance_df.head(10))

print("\nNumber of selected features:", len(top300_features))



#predictions = model.predict(X_test)
# Keep only top 300 features

X_train_top300 = X_train[top300_features]
X_test_top300 = X_test[top300_features]

# Train new RF

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_top300, y_train)

predictions = model.predict(X_test_top300)



print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# SHAP
print("\nRunning SHAP...")

explainer = shap.TreeExplainer(model)

#X_sample = X_test.sample(
 #   n=min(1000, len(X_test)),
  #  random_state=42
#)

X_sample = X_test_top300.sample(
    n=min(1000, len(X_test_top300)),
    random_state=42
)
shap_values = explainer.shap_values(X_sample)

# Binary classification
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample)
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
else:
    shap.summary_plot(shap_values[:, :, 1], X_sample)
    shap.summary_plot(shap_values[:, :, 1], X_sample, plot_type="bar")


import shap

print("SHAP version:", shap.__version__)
print("Expected value:", explainer.expected_value)

print(type(shap_values))

if isinstance(shap_values, list):
    print(shap_values[1].shape)
    print(shap_values[1].min(), shap_values[1].max())
else:
    print(shap_values.shape)
    print(shap_values[:, :, 1].min(), shap_values[:, :, 1].max())