import numpy as np
import random
import math
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
    K-Means++ is a way of initializing K-Means by choosing random starting centers with very specific probabilities.
    (k-mean++: The Advantages of Careful Seeding)
"""

class KmeansPP:
    def __init__(self, data, k, max_iters=100, tol=1e-4):
        """
        Args:
            data (numpy array): data
            k (integer): number of cluster
            max_iter (int, optional): Maximum number of iterations during the training process. Defaults to 100.
            tol (optional): Threshold for accepting small changes between iterations (used to stop the algorithm). Defaults to 1e-4.
        """
        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers = []  
        self.labels = [] 
    
    def initialize_centers(self):
        """ Initialize centroids 
        
        Args:
            data (numpy array): data
            
        Return:
            list of centers
        """
        samples = self.data.shape[0]
        
        self.centers = []
        
        #  Select randomly a point from data as the first cluster center
        self.centers.append(self.data[random.randint(0, samples - 1)])
        # first_center_idx = random.randint(0, samples - 1)
        # self.centers.append(X[first_center_idx])
        
        for i in range(1, self.k):
            distances = []
            for point in self.data:
                # Calculate the distance between the point and the current cluster centers
                # Get the minimum distance
                min_distance = min(np.linalg.norm(point - c) ** 2 for c in self.centers)
                distances.append(min_distance)
                
            # Select the point with the largest distance as the next cluster center
            self.centers.append(self.data[np.argmax(distances)])
            
        return np.array(self.centers)
    
    def predict(self):
        """Assgin label for data points using Euclidean distance

        Args:
            data (numpy array): data
        
        Return: 
            list of data points with labelled cluster
        """
        self.labels = []
        for point in self.data:
            # Calculate the distance between the point and the current cluster centers
            distance_to_cluster = [math.dist(point, c) for c in self.centers]
            
            # Assign the label to the cluster with the closest distance
            self.labels.append(distance_to_cluster.index(min(distance_to_cluster)))
        
        return self.labels
    
    def compute_inertia(self):
        """Measure the clustering quality of the model
            Silhouette Score
        
        """
        inertia = 0
        for i in range(len(self.centers)):
            cluster_points = self.data[np.array(self.labels) == i]
            inertia += np.sum(np.linalg.norm(cluster_points - self.centers[i], axis=1) ** 2)
        return inertia

    def update_centers(self):
        new_centers = []
        for i in range(self.k):
            cluster_points = self.data[np.array(self.labels) == i]
            if len(cluster_points) > 0:
                new_centers.append(np.mean(cluster_points, axis=0))
            else:
                new_centers.append(self.centers[i])
        
        return np.array(new_centers)
    
    def fit(self):
        """ Model training
            1. Initialize the first k cluster centers.
            2. Assign labels to the data points.
            3. Update the cluster centers based on the data points in each cluster.
            4. Check for changes in the cluster centers between iterations.
                If the change in the centers is too small (below the tol threshold), the algorithm will stop.
            5. Repeats until convergence is achieved or the maximum number of iterations is reached.

        Args:
            data (numpy array): data
        """
    
        self.centers = self.initialize_centers()
        prev_centers = np.copy(self.centers)

        for i in range(self.max_iters):
            self.labels = self.predict()
            self.centers = self.update_centers()
            center_shifts = np.linalg.norm(self.centers - prev_centers, axis=1)
            if np.max(center_shifts) < self.tol:
                print(f"Converged at iteration {i+1}")
                break
            prev_centers = np.copy(self.centers)
    
    def save_model(model, file_name):
        """Save the trained model 

        Args:
            model (KMeansPP): 
            file_name (string): file name
        """
        joblib.dump(model, file_name)
    
    def load_model(file_name):
        """Load the saved model

        Args:
            file_name (string, path): file name path
        """
        return joblib.load(file_name)

# -------------------- TEST SECTION --------------------
# if __name__ == "__main__":
    
    # Read data
    # df = pd.read_csv("uk_demographic_customers.csv")
    
    # Get related features
    # data = df[["Recency", "Frequency", "Monetary"]]
    
    # Normalize
    # dt = StandardScaler().fit_transform(data)
    
    # Split data 
    # train_data, test_data = train_test_split(dt, test_size=0.2, random_state=42)
    
    # Training model
    # model = KmeansPP(data=train_data, k=3)
    # model.fit()
    
    # print("Inertia: ", model.compute_inertia())
    
    # Save model
    # model.save_model("KMeans_Plus_Plus_Model.pkl")
    
    
    # Predict
    # new_labels = model.predict()

    
    
