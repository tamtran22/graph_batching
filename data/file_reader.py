import re
import numpy as np

def read_1D_input(
        file_name : str,
        var_dict = {
            'x' : ['x_end', 'y_end', 'z_end'], 
            'edge_index' : ['PareID', 'ID'], 
            'edge_attr' : ['Length', 'Diameter', 'Gene', 'Lobe', 'Vol1-0']
        }
    ):
    r"""Read Output_subject_Amount_St_whole.dat
    Data format
    ID PareID Length Diameter ... Vol1-0 Vol0 Vol1
    -  -      -      -        ... -      -    -
    -  -      -      -        ... -      -    -
    (---------information of ith branch----------)
    -  -      -      -        ... -      -    -
    """
    def _float(str):
        _dict = {'C':0, 'P':0, 'E':0, 'G':0, 'T':1}
        try:
            return float(str)
        except:
            return _dict[str]
    _vectorized_float = np.vectorized(_float)

    file = open(file_name, 'r')
    # Read header
    header = file.readline()
    # Read data
    data = file.read()
    # Done reading file
    file.close()

    # Process header
    vars = header.replace('\n',' ')
    vars = vars.split(' ')
    vars = list(filter(None, vars))

    n_var = len(vars)

    # Process data
    data = data.replace('\n',' ')
    data = data.split(' ')
    data = list(filter(None, data))

    data = np.array(data).reshape((-1, n_var)).transpose()
    data_dict = {}
    for i in range(len(vars)):
        data_dict[vars[i]] = _vectorized_float(data[i])
    
    # Rearange data
    data_dict['x_end'].insert(0, data_dict['x_start'][0])
    data_dict['y_end'].insert(0, data_dict['y_start'][0])
    data_dict['z_end'].insert(0, data_dict['z_start'][0])

    out_dict = {}
    for var in var_dict:
        out_dict[var] = []
        for data_var in var_dict[var]:
            out_dict[var].append(data_dict[data_var])

    out_dict['x'] = np.array(out_dict['x'], dtype=np.float32).transpose()
    out_dict['edge_index'] = np.array(out_dict['edge_index'], dtype=np.int32)
    out_dict['edge_attr'] = np.array(out_dict['edge_attr'], dtype=np.float32)
    return out_dict

def read_1D_output(
        file_name_lambda,
        subject,
        time_id,
        var_name_dict = {
            'pressure' : 'p',
            'flowrate' : 'flowrate'
        }
    ):
    r"""Read data_plt_nd/plt_nd_000time.dat (all time_id)
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
    """
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

        

