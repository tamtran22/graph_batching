import numpy as np
from data import GraphData
from batching import spectral_clustering
from file_reader import read_file_input, read_file_output


if __name__ == '__main__':
    # Read Output_Amount_St_whole.dat.
    var = read_file_input(file_name='./data_test/Output_10081_Amount_St_whole.dat')
    print('Finished reading input file.')
    # Create graph data instance.
    data = GraphData(var['x'], var['edge_index'], var['edge_attr'])
    print('Finished creating graph data instance.')
    # Spectral clustering
    spectral_clustering(
        data=data,
        n_cluster=500
    )
    print('Finish spectral clustering.')
    
