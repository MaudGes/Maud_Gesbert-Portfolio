# Import necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import joblib

# Read the dataset from a CSV file
heart_disease = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

# Split data into features and target label
X = heart_disease[['GenHlth', 'Age', 'Stroke', 'Sex', 'HighChol', 'HighBP', 'Diabetes', 'PhysHlth', 'BMI', 'DiffWalk']]
y = heart_disease["HeartDiseaseorAttack"]

# Split the data into training and test sets with stratification on the target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Define the continuous and binary columns
continuous_columns = ['BMI', 'GenHlth', 'PhysHlth', 'Age']
# The remaining columns are binary: ['Stroke', 'Sex', 'HighChol', 'HighBP', 'Diabetes', 'DiffWalk']

# Create a ColumnTransformer that scales only the continuous features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_columns)
    ],
    remainder='passthrough'  # Leave the binary features unchanged.
)

# Create the CART model with your specified hyperparameters
cart_model = DecisionTreeClassifier(
    min_samples_split=10,  # Minimum samples required to split an internal node
    min_samples_leaf=10,   # Minimum samples required at a leaf node
    max_features=None,     # All features are considered when finding the best split
    max_depth=8,           # Maximum depth of the tree
    criterion='gini'       # Using Gini impurity to measure the quality of a split
)

# Create the pipeline with the preprocessor and the CART model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', cart_model)
])

# Optionally, evaluate the pipeline using stratified 5‑fold cross-validation on the training set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the trained pipeline on the test set
test_accuracy = pipeline.score(X_test, y_test)
print("Test set accuracy:", test_accuracy)

# Export (save) the pipeline using joblib for later deployment in your API
joblib.dump(pipeline, 'heart_disease_cart_pipeline.joblib')
print("Pipeline exported as 'heart_disease_cart_pipeline.joblib'")
