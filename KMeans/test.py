import pandas as pd
from kmeans_model import KMeansModel 

from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv("uk_demographic_customers.csv")

# Get related features
data = df[["Recency", "Frequency", "Monetary"]]

# Normalize
dt = StandardScaler().fit_transform(data)

model = KMeansModel(dt, 3)
model.fit()

model.save_model("kmeans_model.pkl")

print(len(model.labels))
print(model.compute_inertia())

