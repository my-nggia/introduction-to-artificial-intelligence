from kmeans_plus_plus import KmeansPP
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv("uk_demographic_customers.csv")

# Get related features
data = df[["Recency", "Frequency", "Monetary"]]

# Normalize
dt = StandardScaler().fit_transform(data)

# Load model
kmeans = KmeansPP.load_model('KMeans_Plus_Plus_Model.pkl')

# Assgin data
kmeans.data = dt

labels = kmeans.predict()

print(len(labels))