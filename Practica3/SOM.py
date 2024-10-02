# Importar las librerías necesarias
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn_som.som import SOM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

communities_data_file = "communities.data"

# 
communities_data = pd.read_csv(communities_data_file, header=None)

#
communities_data.replace("?", pd.NA, inplace=True)

communities_data_numeric = communities_data.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='mean')
communities_data_imputed = imputer.fit_transform(communities_data_numeric)


print(f"Datos después de la imputación: {communities_data_imputed.shape[0]} filas, {communities_data_imputed.shape[1]} columnas")

# 
scaler = StandardScaler()
communities_data_scaled = scaler.fit_transform(communities_data_imputed)

som = SOM(m=3, n=1, dim=communities_data_scaled.shape[1])


som.fit(communities_data_scaled)


predictions = som.predict(communities_data_scaled)

x = communities_data_imputed[:, 0]
y = communities_data_imputed[:, 1]


colors = ['red', 'green', 'blue']


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))


scatter = ax.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
ax.set_title('SOM Predictions: Communities Data')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')


plt.show()
