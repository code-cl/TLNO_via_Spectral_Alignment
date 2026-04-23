import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import pyvista as pv
import numpy as np
from utils.lapy import TriaMesh, TetMesh, Solver
import matplotlib.pyplot as plt
import scipy.io as sio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    
    name = 'Petals4_data'
    data_path = '../Data/'+name
    
    mesh_data = sio.loadmat(data_path + '/Face_mesh.mat')
    points = mesh_data["nodes"]
    # elements = mesh_data["element"] - 1
    
    # import pyvista as pv
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d(alpha = 1)
    elements = mesh.faces.reshape(-1, 4)[:, 1:]  # 每行第一个数字是顶点数（3表示三角形）
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', opacity=1, show_edges=True)
    plotter.add_axes()
    plotter.show()
    # mesh.plot()
    
    if 1:
        points   = points
        elements = elements
        print(np.min(elements), np.max(elements))
    
        k = 128
        mesh = TriaMesh(points, elements)
        fem  = Solver(mesh)
        evals, evecs  = fem.eigs(k=k)
        stiffness = fem.stiffness
        mass = fem.mass
        stiffness = stiffness.toarray()
        mass = mass.toarray()
        
        evDict = dict()
        evDict['Eigenvalues' ] = evals
        evDict['Eigenvectors'] = evecs
        evDict['Points']    = points
        evDict['Elements' ] = elements
        # evDict['Stiffness' ] = stiffness
        evDict['Mass' ]      = mass
        sio.savemat(data_path + '/Nodes_LBO_basis.mat', evDict)  
    