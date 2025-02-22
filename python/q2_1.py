import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

thetas = np.linspace(0,math.pi, 10000)

x_vals = np.array([10,20,30])
y_vals = np.array([10,20,30])

f, ax = plt.subplots()
for i in range(0,len(x_vals)):
    sns.lineplot(x=thetas, y=x_vals[i]*np.cos(thetas) + y_vals[i]*np.sin(thetas))

plt.show()