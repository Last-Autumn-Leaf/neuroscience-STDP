import numpy as np
import matplotlib.pyplot as plt



a=-.05
b=.075
K=2

x = np.linspace(0, 100, 1000)
y = K* np.exp(a*x)*(np.cos(b*x) - a/b*np.sin(b*x) )

plt.plot(x, y)
plt.grid()
plt.show() # affiche la figure a l'ecran