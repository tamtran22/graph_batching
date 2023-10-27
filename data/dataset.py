import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from data.data import TorchGraphData
from typing import Optional, Callable, Union, List, Tuple
from data.file_reader import *
from preprocessing.batching import get_batch_graphs, merge_graphs
from preprocessing.normalize import normalize_graph, calculate_weight, calculate_derivative


class objectview(object):
    def __init__(self, d) -> None:
        self.__dict__ = d


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
            _filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt', \
                                           'batched_id.pt', 'batched_info.pt']
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
        return self.get(index)
    
    def get(self, index):
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
    _clean_sub_dir() : clear a subfolder.
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
    
    def median(self, var_name=None, axis=None):
        if var_name is None:
            return 0
        else:
            var = []
            for i in range(self.len()):
                data = self.__getitem__(i)
                var.append(data._store[var_name])
            var = torch.cat(var, dim=0)
            if axis is None:
                return var.median()
            else:
                return var.median(axis=axis)

    def batching(self, batch_size : int, batch_n_times : int, recursive : bool, sub_dir='/batched', step=1):
        ''' Perform batching and return batched dataset.
        batch_size : approximate size of sub-graph datas.
        batch_n_times : number of timesteps in sub-graph datas.
        recursive : indicator to partition recursive sub-graphs.
        sub_dir : sub folder to store batched dataset.
        '''
        self._clean_sub_dir(sub_dir=sub_dir)
        os.system(f'mkdir {self.root}{sub_dir}')
        batched_dataset = []
        batched_dataset_id = []
        for i in range(self.len()):
            batched_data = get_batch_graphs(
                data=self.__getitem__(i),
                batch_size=batch_size,
                batch_n_times=batch_n_times,
                recursive=recursive,
                step=step
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
        ''' Map batched data index which is stored in batched dataset to original 
        data index which is stored in raw processed dataset.
        In case of dataset is batched (_sub_dir==/batched), return an array which index is
        the index of batched data (in data_names) and value is the index of parrent data.
        In case of dataset is not batched, return zero tensor.

        '''
        try:
            return torch.load(f'{self.root}{sub_dir}/batched_id.pt')
        except:
            return torch.tensor(0)

    def _clean_sub_dir(self, sub_dir='/batched'):
        ''' Clear the sub folder to store new processed data.
        '''
        if sub_dir == '' or sub_dir == '/':
            print('Unable to clear root folder!')
        else:
            os.system(f'rm -rf {self.root}{sub_dir}/')

    def normalizing(self, sub_dir='/normalized/'):
        ''' Perform normalizing and return normalized dataset.
        sub_dir : sub folder to store normalized dataset.
        '''
        self._clean_sub_dir(sub_dir=sub_dir)
        os.system(f'mkdir {self.root}{sub_dir}')
        # Calculate normalize params
        pressure_min = self.min('pressure')
        pressure_max = self.max('pressure')
        flowrate_min = self.min('flowrate')
        flowrate_max = self.max('flowrate')
        # pressure_dot_min = self.min('pressure_dot')
        # pressure_dot_max = self.max('pressure_dot')
        # flowrate_dot_min = self.min('flowrate_dot')
        # flowrate_dot_max = self.max('flowrate_dot')
        flowrate_bc_min = self.min('flowrate_bc')
        flowrate_bc_max = self.max('flowrate_bc')


        # Cases with node attributes
        node_attr_min = self.min('node_attr', axis=0)
        node_attr_max = self.max('node_attr', axis=0)
        # vol0_min = node_attr_min[-2]
        # vol1_min = node_attr_min[-1]
        # vol0_max = node_attr_max[-2]
        # vol1_max = node_attr_max[-1]
        # node_attr_min[-2] = min(vol0_min, vol1_min)
        # node_attr_min[-1] = min(vol0_min, vol1_min)
        # node_attr_max[-2] = max(vol0_max, vol1_max)
        # node_attr_max[-1] = max(vol0_max, vol1_max)



        kwargs_minmax = objectview({'min':node_attr_min,'max':node_attr_max})
        kwargs_logarithmic = objectview({
            'min':node_attr_min,
            'max':node_attr_max,
            'logscale':torch.tensor([-1,-1,-1,1e2,1e2,-1,-1,-1,1e0,1e0]).float()
        })

        # Normalize
        file_names = [f'{data_name}.pt' for data_name in self.data_names]
        for i in range(self.len()):
            normalized_data = normalize_graph(
                data=self.__getitem__(i),
                node_attr_pipeline=[
                    ([0,1,2,5,6,7],'minmax', kwargs_minmax),
                    ([3,4,8,9],'logarithmic', kwargs_logarithmic)
                ],
                pressure_min=pressure_min, pressure_max=pressure_max, pressure_logscale = 1e6,
                flowrate_min=flowrate_min, flowrate_max = flowrate_max, flowrate_logscale = 1e12,
                # pressure_dot_min=pressure_dot_min, pressure_dot_max=pressure_dot_max,
                # flowrate_dot_min=flowrate_dot_min, flowrate_dot_max = flowrate_dot_max, #flowrate_dot_logscale = 1e10,
                flowrate_bc_min=flowrate_bc_min, flowrate_bc_max=flowrate_bc_max
            )
            # adding weight

            node_weight = calculate_weight(x=normalized_data.node_attr[:,0], bins=10000)
            setattr(normalized_data, 'node_weight', torch.tensor(node_weight, dtype=torch.float32))

            torch.save(normalized_data, f'{self.root}{sub_dir}/{file_names[i]}')
        return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)

    def calculate_derivative(self, 
        sub_dir='/calculated/',
        var_name : Union[List[str], str] = None, 
        axis : int = None, 
        delta_t : float = None
    ):
        if var_name == None:
            return self
        else:
            self._clean_sub_dir(sub_dir=sub_dir)
            os.system(f'mkdir {self.root}{sub_dir}')
            if not isinstance(var_name, list):
                _var_names=[var_name]
            else:
                _var_names=var_name
            for i in range(self.len()):
                data = self.__getitem__(i)
                for _var_name in _var_names:
                    F_dots = calculate_derivative(
                        data=data,
                        var_name=_var_name,
                        axis=axis,
                        delta_t=delta_t
                    )
                    setattr(data, f'{_var_name}_dot', F_dots)
                torch.save(data, f'{self.root}{sub_dir}/{self.data_names[i]}.pt')
            return OneDDatasetLoader(root_dir=self.root, sub_dir=sub_dir)


        




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
    
    def len(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return self.get(index)
    
    def get(self, index):
        return torch.load(self.processed_file_names[index])






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
        CFD_1D_dir = 'CFD_1D'
        # Output_subject_Amout_St_whole.dat
        file_name_input = lambda subject : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/Output_{subject}_Amount_St_whole.dat'
        # data_plt_nd/plt_nd_000time.dat
        file_name_output = lambda subject, time : f'{self.raw}/{subject}'+\
            f'/{CFD_1D_dir}/data_plt_nd/plt_nd_000{time}.dat'
        for subject in self.data_names:
            print(f'Process subject number {self.data_names.index(subject)}, subject name : {subject}.')
            
            data_dict_input = read_1D_input(file_name_input(subject))
            
            file_name_outputs = [file_name_output(subject, time) for time in self.time_id]
            data_dict_output = read_1D_output(file_name_outputs)

            # diam = np.expand_dims(data_dict_input['node_attr'][:,1], axis=1)
            # data_dict_output['velocity'] = data_dict_output['flowrate'] / \
            #             (np.pi * np.square(diam) / 4) # U = Q / A = Q / (pi*d^2/4)
            flowrate_bc = [np.expand_dims(data_dict_output['flowrate'][0,:], axis=0)] \
                            *data_dict_output['flowrate'].shape[0]
            flowrate_bc = np.concatenate(flowrate_bc, axis=0)

            data = TorchGraphData(
                # x = torch.tensor(data_dict_input['node_attr']).type(torch.float32),
                edge_index = torch.tensor(data_dict_input['edge_index']).type(torch.LongTensor),
                edge_attr = torch.tensor(data_dict_input['edge_attr']).type(torch.float32),
                node_attr = torch.tensor(data_dict_input['node_attr']).type(torch.float32),
                pressure = torch.tensor(data_dict_output['pressure']).type(torch.float32),
                flowrate = torch.tensor(data_dict_output['flowrate']).type(torch.float32),
                # velocity = torch.tensor(data_dict_output['velocity']).type(torch.float32)
                flowrate_bc = torch.tensor(flowrate_bc).type(torch.float32)
            )
            torch.save(data, self.processed_file_names[self.data_names.index(subject)])

if __name__=='__main__':
    dataset = OneDDatasetLoader(
        root_dir='/data1/tam/downloaded_datasets_transformed',
        sub_dir='/processed/'
    )