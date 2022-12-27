import re
import numpy as np



##################################################################
##################################################################
# 1D file reader
def read_file_input(
        file_name,
        var_name_dict = {
            'x' : ['x_end', 'y_end', 'z_end'], 
            'edge_index' : ['PareID', 'ID'], 
            'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol1-0']
        }
    ):
    '''
    Read Output_subject_Amount_St_whole.dat
    Data format
    ID PareID Length Diameter ... Vol1-0 Vol0 Vol1
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith branch----------)
    -  -      -      -        ... -      -    -
    '''
    def node_type_to_float(str):
        node_type_dict = {
            'C':0, 'P':0, 'E':0, 'G':0, 'T':1
        }
        try:
            return float(str)
        except:
            return node_type_dict[str]
    file = open(file_name, 'r')
    line = file.readline()[:-1]
    var_names = list(filter(None, line.split(' ')))
    data_dict = {}
    for var_name in var_names:
        data_dict[var_name] = []
    while (True):
        line = file.readline()[:-1]
        if not line:
            break
        vars = list(filter(None, line.split(' ')))
        for var_name, var in zip(var_names, vars):
            data_dict[var_name].append(node_type_to_float(var))
    data_dict['x_end'].insert(0, data_dict['x_start'][0])
    data_dict['y_end'].insert(0, data_dict['y_start'][0])
    data_dict['z_end'].insert(0, data_dict['z_start'][0])
    var_dict = {}
    for var in var_name_dict:
        var_dict[var] = []
        for data_var in var_name_dict[var]:
            var_dict[var].append(data_dict[data_var])
    var_dict['x'] = np.transpose(np.array(var_dict['x'], dtype=np.float32))
    var_dict['edge_index'] = np.array(var_dict['edge_index'], dtype=np.int32)
    var_dict['edge_attr'] = np.transpose(np.array(var_dict['edge_attr'], dtype=np.float32))
    file.close()
    return var_dict

def read_file_output(
        file_name_lambda,
        subject,
        time_id,
        var_name_dict = {
            'pressure' : 'p',
            'flowrate' : 'flowrate' 
        }
    ):
    ''' 
    Read data_plt_nd/plt_nd_000time.dat (all time_id)
    Data format
    VARIABLES="x" "y" "z" "p" ... "flowrate"  "resist" "area"                                    
     ZONE T= "plt_nd_000time.dat                                 "
     N=       xxxxx , E=       xxxxx ,F=FEPoint,ET=LINESEG
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith node----------)
    -  -      -      -        ... -      -    -
    -  -
    -  -
    (---------connectivity of jth branch-------)
    -  -
    '''
    file = open(file_name_lambda(subject, time_id[0]), 'r')
    line = file.readline()[:-1]
    line = list(filter(None, line.split('=')))[1]
    var_names = list(filter(None, line.split(' ')))
    var_names = [var_name[1:-1] for var_name in var_names]
    file.close()
    var_dict = {}
    for var in var_name_dict:
        var_dict[var] = []
        for time in time_id:
            var_dict[var].append([])
    for time in time_id:
        file = open(file_name_lambda(subject, time), 'r')
        file.readline()
        file.readline()
        file.readline()
        while (True):
            line = file.readline()[:-1]
            if not line:
                break
            vars = list(filter(None, line.split(' ')))
            if len(vars) < len(var_names):
                break
            for var_name in var_name_dict:
                index_of_var_name = var_names.index(var_name_dict[var_name])
                value = float(vars[index_of_var_name])
                var_dict[var_name][time_id.index(time)].append(value)
        file.close()
    for var_name in var_name_dict:
        var_dict[var_name] = np.transpose(np.array(var_dict[var_name], dtype=np.float32))
    return var_dict




#############################################################################
#############################################################################
# 3D mesh based file reader

def zone_type(zone_name):
    zone_types = ['iso', 'wall-vol','fan', 'inlet', 'wall', 'vol']
    for _zone_type in zone_types:
        if re.search(_zone_type, zone_name):
            return _zone_type
    return ''

def read_file_tec_face_based(
        file_name, 
        num_var_output = 1
    ):
    '''
    Read tecplot face based - FEPolygon data file
    Data format
    TITLE     = "Translation of CGNS file case_302.cgns"
    VARIABLES = "var-1"
    ...
    "var-num_var"
    ZONE T="zone_name_1"
     ###
     Nodes=##, Faces=##, Elements=##, ZONETYPE=FEPolygon
     DATAPACKING=BLOCK
     ###
     DT=(DOUBLE DOUBLE DOUBLE DOUBLE )
     (node_data) num_var X num_node
     # face nodes
     (face_data) num_face X 2
     # left elements
     (left_elem_data) num_face
     # right elements
     (right_elem_data) num_face
    ZONE T="zone_name_2"
    ...
    '''
    file = open(file_name, 'r')
    file_str = file.read()
    file.close()
    zone = file_str.split('ZONE T=')
    num_var = len(list(filter(None, zone[0].split('\n')))) - 1

    x_wall = []
    for i in range(1, len(zone)):
        header = zone[i].split('DT=')[0]
        header = header.replace('\n',',', header.count('\n'))
        header = header.replace(' ', '', header.count(' '))
        header = header.split(',')
        num_node = 0
        num_face = 0
        num_elem = 0
        zone_name = header[0]
        zone_name = zone_name.replace('"', '', zone_name.count('"'))
        
        for s in header:
            if re.match('^Nodes=', s):
                num_node = int(s.split('=')[-1])
            if re.match('^Faces', s):
                num_face = int(s.split('=')[-1])
            if re.match('^Elements', s):
                num_elem = int(s.split('=')[-1])
        data = zone[i].split('DT=')[1]
        data = data.replace('DOUBLE', ' ', data.count('DOUBLE'))
        data = data.replace('SINGLE', ' ', data.count('SINGLE'))
        data = data.replace('(', ' ', data.count('('))
        data = data.replace(')', ' ', data.count(')'))
        data = data.replace('\n', ' ', data.count('\n'))
        data = data.replace('#', ' ', data.count('#'))
        data = data.replace('face nodes', ' ', data.count('face'))
        data = data.replace('left elements', ' ', data.count('left'))
        data = data.replace('right elements', ' ', data.count('right'))
        data = list(filter(None, data.split()))

        i_start = 0
        i_end = i_start + (num_var - num_var_output)*num_node
        x = np.array(data[i_start:i_end], dtype=np.float32).reshape((num_var-num_var_output, num_node))
        x = x.transpose()

        if zone_type(zone_name) in ['iso']:
            i_start = i_end
            i_end = i_start + num_var_output*num_node
            aoa = np.array(data[i_start:i_end], dtype=np.float32).reshape((num_var_output, num_node))
            aoa = aoa.transpose()

            i_start = i_end
            i_end = i_start + 2*num_face
            edge_index = np.array(data[i_start:i_end], dtype=np.int32).reshape((num_face, 2))
            edge_index = edge_index.transpose() - 1

            edge_attr = np.sqrt(np.sum(np.square(x[edge_index[0]]-x[edge_index[1]]), axis=1)).reshape((num_face,1))
            # edge_index = np.concatenate([edge_index, edge_index], axis=1)
            # edge_attr = np.concatenate([edge_attr,edge_attr],axis=0)

            i_start = i_end
            i_end = i_start + num_face
            left_elem = np.array(data[i_start:i_end], dtype=np.int32)

            i_start = i_end
            i_end = i_start + num_face
            right_elem = np.array(data[i_start:i_end], dtype=np.int32)

            data_dict = {
                'x' : x,
                'age_of_air' : aoa,
                'edge_index' : edge_index,
                'edge_attr' : edge_attr,
                'left_element' : left_elem,
                'right_element' : right_elem,
                'num_elem' : num_elem
            }
        else:
            x_wall.append(x[:,:2])
            
    x_wall = np.concatenate(x_wall)

    id_wall = np.zeros(shape=(data_dict['x'].shape[0], 1), dtype=np.int32)
    mean_mesh_size = np.mean(edge_attr)
    for i in range(x_wall.shape[0]):
        x = data_dict['x'][:,:2]
        d = np.sqrt(np.sum(np.square(x - x_wall[i]), axis=1))
        if np.min(d) <= mean_mesh_size:
            id_wall[np.argmin(d)] = 1
    
    data_dict['x'] = np.concatenate([data_dict['x'], id_wall], axis=1)
    return data_dict
    

if __name__ == '__main__':
    data = read_file_tec_face_based('fan_first_302.dat')
    print(data)
        