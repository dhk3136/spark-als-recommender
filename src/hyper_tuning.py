import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# Read in our data and name the columns appropriately
df = pd.read_csv("../data/3dgraphdata.csv")
df.rename(index=str, columns={"1.000000000000000000e+00": "Rank",
                              "5.000000000000000278e-02": "regParam",
                              "9.211091946857221657e-01": "RMSE"}, inplace=True)


X = df['Rank'].values
Y = df['regParam'].values
Z = df['RMSE'].values

# Plot our data 360 times for each angle
for angle in range(360):
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z)
    ax.set_xlabel("Rank")
    ax.set_ylabel("regParam")
    ax.set_zlabel("RMSE");
    ax.view_init(None, angle)
    if angle < 10:
        title = f"00{angle}.png"
    elif angle < 100:
        title = f"0{angle}.png"
    else:
        title = f"{angle}.png"
    ax.figure.savefig(f"../images/frames/full/{title}")
