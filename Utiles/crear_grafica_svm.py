import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import svm
from sklearn.datasets import make_blobs, make_circles
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURACIÓN DE ESTILO ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 9,
    "figure.autolayout": False 
})

COLOR_0 = '#2C3E50' # Azul oscuro
COLOR_1 = '#E74C3C' # Rojo suave

def plot_svc_decision_boundary(model, ax, plot_support=True):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.7,
               linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=180,
                   linewidth=1.5, facecolors='none', edgecolors='k', label='Vectores de Soporte')

# --- GRÁFICO 1 ---
def plot_linear_external_legend():
    fig, ax = plt.subplots(figsize=(7, 6))
    X, y = make_blobs(n_samples=60, centers=2, random_state=6, cluster_std=0.8)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    ax.scatter(X[y==0][:, 0], X[y==0][:, 1], c=COLOR_0, s=80, edgecolors='white', label='Clase A')
    ax.scatter(X[y==1][:, 0], X[y==1][:, 1], c=COLOR_1, s=80, edgecolors='white', marker='^', label='Clase B')
    plot_svc_decision_boundary(clf, ax)
    ax.set_title("Linealmente Separable (SVM)", fontweight='bold', pad=15)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("SVM_1_Lineal.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# --- GRÁFICO 2 ---
def plot_nonlinear_external_legend():
    fig, ax = plt.subplots(figsize=(7, 6))
    X_circ, y_circ = make_circles(n_samples=120, factor=0.3, noise=0.08, random_state=42)
    ax.scatter(X_circ[y_circ==0][:, 0], X_circ[y_circ==0][:, 1], c=COLOR_0, s=80, edgecolors='white', label='Clase A')
    ax.scatter(X_circ[y_circ==1][:, 0], X_circ[y_circ==1][:, 1], c=COLOR_1, s=80, edgecolors='white', marker='^', label='Clase B')
    x_min, x_max = ax.get_xlim()
    ax.plot([x_min, x_max], [0, 0], 'k--', alpha=0.5, label='Separador Lineal (Fallo)')
    ax.set_title("No Linealmente Separable", fontweight='bold', pad=15)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("SVM_2_NoLineal.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# --- GRÁFICO 3 (SOLUCIÓN DEFINITIVA DE ESPACIO) ---
def plot_rbf_3d_full_space():
    fig = plt.figure(figsize=(9, 8)) # Un poco más ancho para compensar la izquierda
    ax = fig.add_subplot(111, projection='3d')
    
    X_circ, y_circ = make_circles(n_samples=100, factor=0.3, noise=0.08, random_state=42)
    gamma = 1.5
    Z = np.exp(-gamma * np.sum(X_circ**2, axis=1))
    
    # Puntos
    ax.scatter(X_circ[y_circ==0][:, 0], X_circ[y_circ==0][:, 1], Z[y_circ==0], 
               c=COLOR_0, s=60, edgecolors='k', linewidth=0.5, alpha=0.8)
    ax.scatter(X_circ[y_circ==1][:, 0], X_circ[y_circ==1][:, 1], Z[y_circ==1], 
               c=COLOR_1, s=60, edgecolors='k', linewidth=0.5, marker='^', alpha=0.8)

    # Plano
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 20), np.linspace(-1.2, 1.2, 20))
    zz = np.full_like(xx, 0.4)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='black')
    
    # Títulos y etiquetas
    ax.set_title("Proyección Kernel RBF (3D)", fontweight='bold', pad=20)
    ax.set_xlabel("$x_1$", labelpad=10)
    ax.set_ylabel("$x_2$", labelpad=10)
    
    # Aquí está el truco para la etiqueta Z: labelpad más grande
    ax.set_zlabel("$\phi(x)$", labelpad=15)
    
    # Vista
    ax.view_init(elev=25, azim=35)
    
    # AJUSTE MANUAL DE MÁRGENES
    # Le damos mucho espacio a la izquierda (0.15) y abajo (0.1)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95)
    
    # pad_inches=0.2 asegura un marco blanco extra alrededor de todo
    plt.savefig("SVM_3_RBF_3D_Full.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

if __name__ == "__main__":
    plot_linear_external_legend()
    plot_nonlinear_external_legend()
    plot_rbf_3d_full_space()
    print("Gráficas generadas exitosamente con márgenes corregidos.")