import os
from typing import Optional, Callable, Union, List, Tuple
import numpy as np
import torch
import time
from torch_geometric.data import InMemoryDataset, Data
from preprocessing.file_reader import *
from preprocessing.data import TorchGraphData



##########################################################################
##########################################################################


    
def transform_scaler(
        data : Data,
        transform_var_names : dict
    ):
    # data component: x, edge_index, edge_attr, pressure, flowrate
    data_transformed = data
    for var_name in transform_var_names:
        transform_type = transform_var_names[var_name]
        var = getattr(data, var_name)
        if transform_type == 'standard':
            _mean = torch.mean(var, dim=0)
            _std = torch.std(var, dim=0)
            var = (var - _mean) / (_std + 1e-15)
        elif transform_type == 'min_max':
            _min = torch.min(var, dim=0).values
            _max = torch.max(var, dim=0).values
            var = (var - _min) / (_max - _min)
        elif transform_type == 'robust':
            _median = torch.median(var, dim=0).values
            _percentile75 = torch.quantile(var, q=0.75, dim=0)
            _percentile25 = torch.quantile(var, q=0.25, dim=0)
            var = (var - _median) / (_percentile75 - _percentile25 + 1e-15)
        else:
            print(f'{transform_type} transform type is not available.')
        setattr(data_transformed, var_name, var)
    return data_transformed



#################################################################
#################################################################
class BaseGraphDataset(InMemoryDataset):
    def __init__(self,
        raw_file_dir: Optional[str] = None,
        root_dir: Optional[str] = None, 
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None,
        is_loader: Optional[bool] = False
    ):
        # self.print_doc()
        self.raw_file_dir = raw_file_dir
        self.is_loader = is_loader
        super(BaseGraphDataset, self).__init__(root_dir, transform, pre_transform, pre_filter)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        return [self.root+'/processed/'+data+'.pt' 
                                for data in self.data_names]

    def process(self, **kwargs):
        raise NotImplementedError

    @property
    def len(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return torch.load(self.processed_file_names[index])

    def print_doc(self):
        if self.is_loader:
            print('Loader for existed OneDGraph dataset.')
            print('Requirements:')
            print('    root_dir')
        else:
            print('Processor for new OneDGraph dataset.')
            print('Requirements:')
            print('    raw_file_dir')
            print('    root_dir')






##################################################################
##################################################################
class OneDGraphDataset(BaseGraphDataset):
    def __init__(self, 
        raw_file_dir: Optional[str] = None,
        root_dir: Optional[str] = None, 
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        time_id: Union[List[str], Tuple] = None,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None, 
        is_loader: Optional[bool] = False
    ):
        self._data_names = data_names
        self.time_id = time_id
        super(OneDGraphDataset, self).__init__(raw_file_dir, root_dir, transform, pre_transform, pre_filter, is_loader)

    @property
    def data_names(self):
        if self._data_names == 'all':
            if self.is_loader:
                data_dir = self.root + '/processed/'
                data_names = os.listdir(data_dir)
                _filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt']
                data_names = list(filter(_filter, data_names))
                return [data.replace('.pt','',data.count('.pt')) for data in data_names]
            else:
                data_dir = self.raw_file_dir
                data_names =  os.listdir(data_dir)
                return data_names
        else:
            return self._data_names
    
    def process(self):
        if self.is_loader:
            print('Dataset loader passes processing input data.')
            pass
        else:
            # Output_subject_Amout_St_whole.dat
            file_name_input = lambda subject : self.raw_file_dir+'/'+subject+\
                '/CFD_1D/Output_'+subject+'_Amount_St_whole.dat'
            # data_plt_nd/plt_nd_000time.dat
            file_name_output = lambda subject, time : self.raw_file_dir+'/'+subject+\
                '/CFD_1D/data_plt_nd/plt_nd_000'+time+'.dat'
            for subject in self.data_names:
                data_dict_input = read_file_input(file_name_input(subject))
                data_dict_output = read_file_output(file_name_output, subject, self.time_id)
                data = TorchGraphData(
                    x = torch.tensor(data_dict_input['x']).type(torch.float32),
                    edge_index = torch.tensor(data_dict_input['edge_index']).type(torch.LongTensor),
                    edge_attr = torch.tensor(data_dict_input['edge_attr']).type(torch.float32),
                    pressure = torch.tensor(data_dict_output['pressure']).type(torch.float32),
                    flowrate = torch.tensor(data_dict_output['flowrate']).type(torch.float32)
                )
                torch.save(data, self.processed_file_names[self.data_names.index(subject)])


                

        
#################################s#################################
##################################################################
class Mesh3DFaceBasedGraphDataset(BaseGraphDataset):
    def __init__(self, 
        raw_file_dir: Optional[str] = None, 
        root_dir: Optional[str] = None, 
        data_names: Optional[Union[str, List[str], Tuple]] = 'all',
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None, 
        is_loader: Optional[bool] = False
    ):
        self._data_names = data_names
        super().__init__(raw_file_dir, root_dir, transform, pre_transform, pre_filter, is_loader)
    
    @property
    def data_names(self) -> Union[List[str], Tuple]:
        if self._data_names == 'all':
            if self.is_loader:
                data_dir = self.root + '/processed/'
                data_names = os.listdir(data_dir)
                _filter = lambda s : not s in ['pre_filter.pt', 'pre_transform.pt']
                data_names = list(filter(_filter, data_names))
                return [data.replace('.pt','',data.count('.pt')) for data in data_names]
            else:
                data_dir = self.raw_file_dir
                data_names =  os.listdir(data_dir)
                return [data.replace('.dat','',data.count('.dat')) for data in data_names]
        else:
            return self._data_names

    def process(self):
        if self.is_loader:
            print('Dataset loader passes processing input data.')
            pass
        else:
            if os.path.exists(self.root):
                os.system('rm -rf '+self.root+'/processed/*')
            file_name = lambda case : self.raw_file_dir+'/'+case+'.dat'
            for case in self.data_names:
                print(f'Process case number {self.data_names.index(case)}, case name {case}.')
                data_dict = read_file_tec_face_based(file_name(case))
                data = Data(
                    x = torch.tensor(data_dict['x']).type(torch.float32),
                    edge_index=torch.tensor(data_dict['edge_index']).type(torch.LongTensor),
                    edge_attr=torch.tensor(data_dict['edge_attr']).type(torch.float32),
                    age_of_air = torch.tensor(data_dict['age_of_air']).type(torch.float32),
                    left_element=torch.tensor(data_dict['left_element']).type(torch.LongTensor),
                    right_element=torch.tensor(data_dict['right_element']).type(torch.LongTensor),
                    num_elem=data_dict['num_elem']
                )
                torch.save(data, self.processed_file_names[self.data_names.index(case)])
##################################################################
##################################################################

def transform_wrapper(data):
    transform_type = {
        'x' : 'standard',
        'edge_attr' : 'min_max',
        'pressure' : 'robust',
        'flowrate' : 'robust'
    }
    return transform_scaler(data, transform_type)

if __name__ == '__main__':
    current_time = time.time()
    dataset = Mesh3DFaceBasedGraphDataset(
        raw_file_dir='./',
        root_dir='/data1/tam/downloaded_datasets_test',
        data_names=['fan_first_302'],
        # time_id=[str(i).zfill(3) for i in range(201)],
        transform=None,
        pre_transform=None,
        pre_filter=None,
        is_loader=False
    )
    print(f"Create dataset in {time.time() - current_time} seconds.")
    # print(dataset[0].x)
    # print(dataset[0].edge_attr)
    # print(dataset[0].flowrate)
    # print(dataset[0].pressure)

