import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from data.data import TorchGraphData
from typing import Optional, Callable, Union, List, Tuple
from data.file_reader import *
from preprocessing.batching import get_batch_graphs, merge_graphs
from preprocessing.normalize import normalize_graph, calculate_weight
from data.file_reader import edge_to_node





class DatasetLoader(Dataset):
    r"""Base dataset loader class for graph data
    Loader is specifically used for loading a processed dataset.
    Function 'process()' is not implemented.

    --properties--
    root_dir : original folder to store dataset.
    _sub_dir : subfolder which stored data for different types of processed data
                '/processed' : original processed datas read by builder.
                '/normalized' : normalized datas (process described in function).
                '/batched' : batched datas which are divided by graph partitioning
                            time step slicing.
    data_names : list of all subject's names, subject's data may be 
                contained in multiple files.
    processed_file_names : list of all processed subject's names, each 
                subject's data is contained in 1 file (.pt).
    len() : number of subjects.


    --methods--
    __getitem__(index) : return data with given index.
    """
    def __init__(self, 
        root_dir: Optional[str] = None, 
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        sub_dir: Optional[str] = None,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None
    ):
        self._data_names = data_names
        if sub_dir is None:
            self._sub_dir = '/processed'
        else:
            self._sub_dir = sub_dir
        super().__init__(root_dir, transform, pre_transform, pre_filter)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        if self._data_names == 'all':
            data_dir = self.root + self._sub_dir + '/'
            data_names = os.listdir(data_dir)
            _filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', 'batched_id.pt', 'batched_info.pt']
            data_names = list(filter(_filter, data_names))
            return [data.replace('.pt','',data.count('.pt')) for data in data_names]
        else:
            return self._data_names

    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        return [self.root+self._sub_dir+'/'+data+'.pt' 
                                for data in self.data_names]

    def process(self):
        pass
    
    def len(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return torch.load(self.processed_file_names[index])





class OneDDatasetLoader(DatasetLoader):
    r"""
    Loader and pre-processing for OneD dataset.

    --properties--
    batching_id : mapping batched index to original index (for batched dataset).

    --methods--
    min() : return minimum of a variable on whole dataset.
    max() : return maximum of a variable on whole dataset.
    mean() : return mean of a variable on whole dataset.
    std() : return standard deviation of a variable on whole dataset.
    batching() : perform batching for all datas in dataset and return the 
                batched dataset.
    _clean_sub_dir : clear a subfolder.
    normalizing() : perform normalizing for all datas in dataset and return 
                the normalized dataset.
    """
    def __init__(self,
        root_dir: Optional[str] = None, 
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        sub_dir: Optional[str] = None,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None
    ):
        super().__init__(root_dir, data_names, sub_dir, transform, pre_transform, pre_filter)
    
    def min(self, var_name=None, axis=None):
        if var_name is None:
            return 0
        else:
            var = []
            for i in range(self.len()):
                data = self.__getitem__(i)
                var.append(data._store[var_name])
            var = torch.cat(var, dim=0)
            if axis is None:
                return var.min()
            else:
                return var.min(axis=axis).values
    
    def max(self, var_name=None, axis=None):
        if var_name is None:
            return 0
        else:
            var = []
            for i in range(self.len()):
                data = self.__getitem__(i)
                var.append(data._store[var_name])
            var = torch.cat(var, dim=0)
            if axis is None:
                return var.max()
            else:
                return var.max(axis=axis).values
    
    def mean(self, var_name=None, axis=None):
        if var_name is None:
            return 0
        else:
            var = []
            for i in range(self.len()):
                data = self.__getitem__(i)
                var.append(data._store[var_name])
            var = torch.cat(var, dim=0)
            if axis is None:
                return var.mean()
            else:
                return var.mean(axis=axis)
    
    def std(self, var_name=None, axis=None):
        if var_name is None:
            return 0
        else:
            var = []
            for i in range(self.len()):
                data = self.__getitem__(i)
                var.append(data._store[var_name])
            var = torch.cat(var, dim=0)
            if axis is None:
                return var.std()
            else:
                return var.std(axis=axis)
        
    def batching(self, batch_size : int, batch_n_times : int, recursive : bool, sub_dir='/batched'):
        self._clean_sub_dir(sub_dir=sub_dir)
        os.system(f'mkdir {self.root}{sub_dir}')
        batched_dataset = []
        batched_dataset_id = []
        for i in range(self.len()):
            batched_data = get_batch_graphs(
                data=self.__getitem__(i),
                batch_size=batch_size,
                batch_n_times=batch_n_times,
                recursive=recursive
            )
            batched_dataset += batched_data
            batched_dataset_id += [i]*len(batched_data)
            
        for i in range(len(batched_dataset)):
            torch.save(batched_dataset[i], f'{self.root}{sub_dir}/batched_data_{i}.pt')
        
        torch.save(torch.tensor(batched_dataset_id), f'{self.root}{sub_dir}/batched_id.pt')
        torch.save({'batch_size' : batch_size, 'batch_n_times':batch_n_times}, f'{self.root}{sub_dir}/batched_info.pt')
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)
    
    @property
    def batching_id(self, sub_dir='/batched'):
        try:
            return torch.load(f'{self.root}{sub_dir}/batched_id.pt')
        except:
            return torch.tensor(0)

    def _clean_sub_dir(self, sub_dir='/batched'):
        if sub_dir == '' or sub_dir == '/':
            print('Unable to clear root folder!')
        else:
            os.system(f'rm -rf {self.root}{sub_dir}/')

    def normalizing(self, sub_dir='/normalized/'):
        self._clean_sub_dir(sub_dir=sub_dir)
        os.system(f'mkdir {self.root}{sub_dir}')
        # Calculate normalize params
        pressure_min = self.min('pressure')
        pressure_max = self.max('pressure')
        velocity_min = self.min('velocity')
        velocity_max = self.max('velocity')
        edge_attr_min = self.min('edge_attr', axis=0)
        edge_attr_max = self.max('edge_attr', axis=0)
        vol0_min = edge_attr_min[-2]
        vol1_min = edge_attr_min[-1]
        vol0_max = edge_attr_max[-2]
        vol1_max = edge_attr_max[-1]
        edge_attr_min[-2] = min(vol0_min, vol1_min)
        edge_attr_min[-1] = min(vol0_min, vol1_min)
        edge_attr_max[-2] = max(vol0_max, vol1_max)
        edge_attr_max[-1] = max(vol0_max, vol1_max)
        # Normalize
        file_names = [f'{data_name}.pt' for data_name in self.data_names]
        for i in range(self.len()):
            # print(f'Dataset {i}')
            normalized_data = normalize_graph(
                data=self.__getitem__(i),
                edge_attr_min=edge_attr_min, edge_attr_max=edge_attr_max,
                pressure_min=pressure_min, pressure_max=pressure_max,
                velocity_min=velocity_min, velocity_max=velocity_max
            )
            # adding weight
            edge_weight = calculate_weight(x=normalized_data.edge_attr[:,0], bins=100)
            setattr(normalized_data, 'edge_weight', torch.tensor(edge_weight, dtype=torch.float32))
            node_weight = edge_to_node(edge_weight, normalized_data.edge_index.numpy())
            setattr(normalized_data, 'node_weight', torch.tensor(node_weight, dtype=torch.float32))

            torch.save(normalized_data, f'{self.root}{sub_dir}/{file_names[i]}')
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)
    
    def add_weight(self, var_name:str = 'edge_attr', dim : int = 0):
        pass


        




#####################################################################
class DatasetBuilder(Dataset):
    r"""Base dataset builder class for graph data
    Builder is specifically used for read/process/save graph dataset. 
    """
    def __init__(self,
        raw_dir: Optional[str] = None,
        root_dir: Optional[str] = None,
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
    ):
        # No transforming for builder class.
        transform = None
        pre_transform = None
        pre_filter = None
        self.raw = raw_dir
        self._data_names = data_names
        super().__init__(root_dir, transform, pre_transform, pre_filter)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        raise NotImplemented

    def process(self):
        # Write code for data processing here.
        raise NotImplemented






class OneDDatasetBuilder(DatasetBuilder):
    r"""Constructing OneD dataset
    """

    def __init__(self, 
        raw_dir: Optional[str] = None, 
        root_dir: Optional[str] = None, 
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        time_id: List[str] = None
    ):
        self.time_id = time_id
        super().__init__(raw_dir, root_dir, data_names)

    @property
    def data_names(self) -> Union[List[str], Tuple]:
        if self._data_names == 'all':
            data_dir = self.raw
            data_names = os.listdir(data_dir)
            # _filter = lambda s : os.path.isdir(self.raw + s)
            # data_names = list(filter(_filter, data_names))
            return data_names
        else:
            return self._data_names
    
    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        return [self.root+'/processed/'+data+'.pt' 
                                for data in self.data_names]

    def process(self):

        # Output_subject_Amout_St_whole.dat
        file_name_input = lambda subject : self.raw+'/'+subject+\
            '/CFD_1D/Output_'+subject+'_Amount_St_whole.dat'
        # data_plt_nd/plt_nd_000time.dat
        file_name_output = lambda subject, time : self.raw+'/'+subject+\
            '/CFD_1D/data_plt_nd/plt_nd_000'+time+'.dat'
        for subject in self.data_names:
            print(f'Process subject number {self.data_names.index(subject)}, subject name : {subject}.')
            
            data_dict_input = read_1D_input(file_name_input(subject))
            
            file_name_outputs = [file_name_output(subject, time) for time in self.time_id]
            data_dict_output = read_1D_output(file_name_outputs)

            diam = np.expand_dims(data_dict_input['edge_attr'][:,1], axis=1)
            data_dict_output['velocity'] = data_dict_output['flowrate'] / \
                        (np.pi * np.square(diam) / 4) # U = Q / A = Q / (pi*d^2/4)

            data = TorchGraphData(
                x = torch.tensor(data_dict_input['node_attr']).type(torch.float32),
                edge_index = torch.tensor(data_dict_input['edge_index']).type(torch.LongTensor),
                edge_attr = torch.tensor(data_dict_input['edge_attr']).type(torch.float32),
                pressure = torch.tensor(data_dict_output['pressure']).type(torch.float32),
                flowrate = torch.tensor(data_dict_output['flowrate']).type(torch.float32),
                velocity = torch.tensor(data_dict_output['velocity']).type(torch.float32)
            )
            torch.save(data, self.processed_file_names[self.data_names.index(subject)])