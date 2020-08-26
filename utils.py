import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import numpy as np


def Lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations"""
    x, y, z = X
    dx = -sigma * (x - y)
    dy = rho*x - y - x*z
    dz = -beta*z + x*y
    return (dx, dy, dz)


def simulate_Lorenz(sigma, beta, rho, x0, y0, z0, tmax, n):
    # Integrate the Lorenz equations on the time grid t.
    t = np.linspace(0, tmax, n)
    f = odeint(Lorenz, (x0, y0, z0), t, args=(sigma, beta, rho))
    x, y, z = f.T
    return x,y,z

def Rossler(X, t, a, b, c):
    """The Rossler equations"""
    x, y, z = X
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + x*z - c*z
    return (x_dot, y_dot, z_dot)

def simulate(equations, a, b, c, x0, y0, z0, tmax, n):
    # Integrate equations on the time grid t.
    t = np.linspace(0, tmax, n)
    f = odeint(equations, (x0, y0, z0), t, args=(a, b, c))
    x, y, z = f.T
    return x,y,z


def plot_trajectories(x,y,z,title, type_plot):
    # Plot the attractor using a Matplotlib 3D projection.
    fig = plt.figure(figsize=(5,3))

    ax = Axes3D(fig)
    
    if type_plot == 'line':
        ax.plot(x, y, z, 'b-', lw=0.5, color = 'lightblue')
    
    if type_plot == 'scatter':
        ax.scatter(x, y, z, s=0.5)

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    plt.tick_params(labelsize=5)
    ax.set_title(title, fontsize=10)

    plt.show()
    
#Downsample data

def voxel_down_sample(data, voxel_size):
    '''
    Down sample the point cloud according to a grid of fixed size.
    INPUT: 
    - data: point cloud, subset of a R^n
    - voxel_size: float number indicating the size of the cubical grid of R^n
    OUTPUT: new point cloud obtained from data by computing the mean of the points of the original point cloud points that are inside each grid cube.
    '''
    dim = len(data[0])
    m = {} #min of the data in each dimension
    for i in range(dim):
        m[i] = data[:,i].min()
    grid_data = {}
    for x in data:
        pos = {}
        for i in range(dim):
          pos[i] = int((x[i]-m[i])/voxel_size)
        tuple_pos = tuple(pos.values())
        if tuple_pos not in grid_data.keys():
          grid_data[tuple_pos] = np.array([x])
        else:
          grid_data[tuple_pos] = np.append(grid_data[tuple_pos], [x], axis = 0)
    mean_grid_data = dict(map(lambda t: (t[0], t[1].mean(axis = 0)), grid_data.items()))
    return np.array(list(mean_grid_data.values()))

def delay_embedding(s, T):
    n=0
    N=len(s)
    step = 1
    xs = s[n:N-2*T:step]
    ys = s[n+T:N-T:step]
    zs = s[n+2*T:N:step]
    
    return xs, ys, zs

from fermat import Fermat
from scipy.spatial import  distance_matrix

def compute_fermat_distances(data, alpha):
    
    #Compute euclideandistances
    distances = distance_matrix(data,data)
    
    # Initialize the model
    fermat = Fermat(alpha = alpha, path_method='FW') 

    # Fit
    fermat.fit(distances)
    
    ##Compute Fermat distances
    fermat_dist = fermat.get_distances()
    
    return  fermat_dist