import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definição da diretriz C(u)
def C(u):
    return np.vstack((u, np.sin(u), np.zeros_like(u))).T

# Definição do vetor de direção d(u)
def d(u):
    return np.array([0, 0, 1])

# Parametrização da superfície regrada
def X(u, v):
    return C(u) + v * d(u)

# Parâmetros de u e v
ui, uf = 0, 2*np.pi
vi, vf = -1, 1

# Geração dos pontos da superfície
u_values = np.linspace(ui, uf, 100)
v_values = np.linspace(vi, vf, 10)
U, V = np.meshgrid(u_values, v_values)
X_values = np.array([X(u, v) for u, v in zip(np.ravel(U), np.ravel(V))])
X_values = X_values.reshape((10, 100, 3))  # Corrigido para a forma correta

# Plotagem da superfície
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_values[:,:,0], X_values[:,:,1], X_values[:,:,2], alpha=0.5)
ax.plot(C(u_values)[:,0], C(u_values)[:,1], C(u_values)[:,2], color='r', label='Diretriz')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
