import numpy as np
import matplotlib.pyplot as plt



a=-.05
b=.075
K=1

x = np.linspace(0, 100, 1000)
y = K* np.exp(a*x)*(np.cos(b*x) - a/b*np.sin(b*x) )
plt.plot(x, y)

G=1
w_n=.05
z=0.7
c=-z*w_n
phi=np.pi/2
y = G*np.exp(c*x)*np.sin(w_n*x+phi)
# G*exp(c*x)*sin(w_n*x+phi)
plt.plot(x, y)
plt.grid()
plt.show() # affiche la figure a l'ecran

#----------------------------------------------------------


# B forme 1
A=1
B=.1
tau=60
tauB=120

x1 = np.linspace(-170, 0, 1000)
x2 = np.linspace(1, 170, 1000)
y1= A*np.exp(x1/tau)
y2= B *np.log(x2/tauB)
plt.plot(x1, y1)
plt.plot(x2, y2)

plt.grid()
plt.show() # affiche la figure a l'ecran