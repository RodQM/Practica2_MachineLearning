
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

adult_data = pd.read_csv("adult.data", names=["age", "workclass", "fnlwgt", "education", "education-num", 
                                              "marital-status", "occupation", "relationship", "race", 
                                              "sex", "capital-gain", "capital-loss", "hours-per-week", 
                                              "native-country", "income"])


adult_data_numeric = adult_data[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]


scaler = StandardScaler()
adult_data_scaled = scaler.fit_transform(adult_data_numeric)

pca = PCA(n_components=2)


pca_result = pca.fit_transform(adult_data_scaled)


pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title('PCA Results: Adult Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
