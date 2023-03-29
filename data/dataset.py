import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from data.data import TorchGraphData
from typing import Optional, Callable, Union, List, Tuple
from data.file_reader import *
from preprocessing.batching_v2 import get_batch_graphs


class DatasetLoader(Dataset):
    r"""Base dataset loader class for graph data
    Loader is specifically used for loading a processed dataset.
    Function 'process()' is not implemented.
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
            self._sub_dir = '/processed/'
        else:
            self._sub_dir = sub_dir
        super().__init__(root_dir, transform, pre_transform, pre_filter)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        if self._data_names == 'all':
            data_dir = self.root + self._sub_dir
            data_names = os.listdir(data_dir)
            _filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', 'batched_id.pt']
            data_names = list(filter(_filter, data_names))
            return [data.replace('.pt','',data.count('.pt')) for data in data_names]
        else:
            return self._data_names

    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        return [self.root+self._sub_dir+data+'.pt' 
                                for data in self.data_names]

    def process(self):
        pass
    
    def len(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return torch.load(self.processed_file_names[index])


class OneDDatasetLoader(DatasetLoader):
    r"""
    aaaaaa
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
        
    def batching(self, batch_size : int, batch_n_times : int, recursive : bool, sub_dir='/processed/'):
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
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)
    
    def get_batching_id(self, sub_dir='/batched/'):
        return torch.load(f'{self.root}{sub_dir}/batched_id.pt')

    def _clean_sub_dir(self, sub_dir='/batched/'):
        if sub_dir == '' or sub_dir == '/':
            print('Unable to clear root folder!')
        else:
            os.system(f'rm -rf {self.root}{sub_dir}')

    def normalizing(self):
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