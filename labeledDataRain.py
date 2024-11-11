import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Define the dataset based on the problem description
# Assuming the table is as follows for illustration (replace with actual data from the assignment):
data = {
    "Humidity": ["High", "Low", "High", "High", "Low"],
    "Cloud_Cover": ["Cloudy", "Sunny", "Sunny", "Cloudy", "Cloudy"],
    "Wind_Strength": ["Medium", "Strong", "Weak", "Medium", "Strong"],
    "Sky_Condition": ["Bright", "Dull", "Bright", "Dull", "Bright"],
    "Rain_Status": ["Rain", "No Rain", "Rain", "Rain", "No Rain"]
}

# Step 2: Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 3: Encode categorical variables
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# Step 4: Split features and target
X = df.drop("Rain_Status", axis=1)
y = df["Rain_Status"]

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Naive Bayes classifier
model = CategoricalNB()
model.fit(X_train, y_train)

# Step 7: Define the input conditions for prediction
# High Humidity, Cloudy Skies, Medium Wind, Bright Sky
input_conditions = np.array([[encoder.transform(["High"])[0],
                              encoder.transform(["Cloudy"])[0],
                              encoder.transform(["Medium"])[0],
                              encoder.transform(["Bright"])[0]]])

# Step 8: Predict the likelihood of rain
predicted_class = model.predict(input_conditions)[0]
predicted_prob = model.predict_proba(input_conditions)[0]

# Step 9: Decode the results back to labels
rain_status = encoder.inverse_transform([predicted_class])[0]
rain_likelihood = predicted_prob[encoder.transform(["Rain"])[0]] * 100


# Step 10: Print the result
print(f"Predicted Status: {rain_status}")
print(f"Likelihood of Rain: {rain_likelihood:.2f}%")
