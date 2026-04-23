import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
from utils.lapy import TriaMesh,Solver
import matplotlib.pyplot as plt
import scipy.io as sio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    
    name      = 'Geo_A'
    figs_path = 'figs/' + name
    data_path = '../Data/'+name
    
    data = sio.loadmat(data_path +'/data')
    Points   = data['MeshNodes'].T
    Points = np.hstack((Points, np.zeros(Points.shape[0]).reshape(-1,1)))   
    elements = (data['MeshElements'].T - 1).astype(int)

    k = 128
    mesh = TriaMesh(Points, elements)
    fem  = Solver(mesh)
    evals, evecs  = fem.eigs(k=k)
    stiffness = fem.stiffness
    mass = fem.mass
    stiffness = stiffness.toarray()
    mass = mass.toarray()
    
    evDict = dict()
    evDict['Eigenvalues' ] = evals
    evDict['Eigenvectors'] = evecs
    evDict['Points']    = Points
    evDict['Elements' ] = elements
    # evDict['Stiffness' ] = stiffness
    evDict['Mass' ]      = mass
    sio.savemat(data_path + '/Nodes_LBO_basis.mat', evDict)  
    