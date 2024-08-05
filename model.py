# model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Use the correct relative or absolute path to your CSV file
df = pd.read_csv('Online Sales Data.csv', encoding='latin1')

# Load and preprocess the data
# df = pd.read_csv(r"C:\Users\archa\OneDrive\Desktop\internship\Task1\Online Sales Data.csv", encoding='latin1')
df = pd.read_csv(r"C:\Users\archa\OneDrive\Desktop\internship\Task1\my_flask_project\Online Sales Data.csv", encoding='latin1')

df.drop(['Transaction ID', 'Date', 'Product Name'], axis=1, inplace=True)
x = df.drop('Total Revenue', axis=1)
y = df['Total Revenue']

# Define features
numeric_features = ['Units Sold', 'Unit Price']
categorical_features = ['Product Category', 'Region', 'Payment Method']

# Define transformations
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42, test_size=0.2)

# Train the best model (Gradient Boosting)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])
model.fit(xtrain, ytrain)

def predict(data):
    # Ensure columns match exactly
    data = data[['Units Sold', 'Unit Price', 'Product Category', 'Region', 'Payment Method']]
    return model.predict(data)
