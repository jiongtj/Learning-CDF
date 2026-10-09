 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

from utils.data_loader import load_mesh_data, load_streams_data
from utils.data_utils import get_projected_vectors


def collate_fn(batch_data):

    fid_num_list = [[single_data.shape[0] for single_data in data['fid']] for data in batch_data]
    sum_fid_num_list = [sum(num) for num in fid_num_list]
    max_fid_num = max(sum_fid_num_list)

    num_points = batch_data[0]['v'].shape[0]

    total_names = list()
    total_v = list()
    total_vn = list()
    total_f = list()
    total_fn = list()
    total_cd1 = list()
    total_cd2 = list()
    total_pv = list()

    total_fid = list()
    total_streams_vec = list()
    total_v_mark = list()

    for batch, data in enumerate(batch_data):

        total_names.append(data['name'])
        total_v.append(data['v'])
        total_vn.append(data['vn'])
        total_f.append(data['f'])
        total_fn.append(data['fn'])
        total_cd1.append(data['cd1'])
        total_cd2.append(data['cd2'])
        total_pv.append(data['pv'])

        temp_fid = np.full(max_fid_num, -1, dtype=np.int32)
        temp_fid[0:sum_fid_num_list[batch]] = np.concatenate(data['fid'], axis=0)
        total_fid.append(temp_fid)

        temp_streams_vec = np.zeros((max_fid_num, 3), dtype=np.float32)
        temp_streams_vec[0:sum_fid_num_list[batch], :] = np.concatenate(data['s_vec'], axis=0)
        total_streams_vec.append(temp_streams_vec)

        temp_v_mark = np.full(num_points, 0.0, dtype= np.float32)
        point_index = data['f'][temp_fid[0:sum_fid_num_list[batch]]].reshape(-1)
        temp_v_mark[point_index] = 1.0
        total_v_mark.append(temp_v_mark)

    total_v = torch.from_numpy(np.array(total_v))
    total_vn = torch.from_numpy(np.array(total_vn))
    total_f = torch.from_numpy(np.array(total_f, dtype=np.int64))
    total_fn = torch.from_numpy(np.array(total_fn))
    total_cd1 = torch.from_numpy(np.array(total_cd1))
    total_cd2 = torch.from_numpy(np.array(total_cd2))
    total_pv = torch.from_numpy(np.array(total_pv))
    total_fid = torch.from_numpy(np.array(total_fid, dtype=np.int64))
    total_streams_vec = torch.from_numpy(np.array(total_streams_vec))
    total_v_mark = torch.from_numpy(np.array(total_v_mark))

    return ({ 'names': total_names, 'v': total_v, 'vn': total_vn, 'f': total_f, 'fn': total_fn,
            'cd1': total_cd1, 'cd2': total_cd2, 'pv': total_pv,
            'fid': total_fid, 's_vec': total_streams_vec, 'v_mark': total_v_mark})


class TriMeshDataset(Dataset):

    def __init__(self, config, dataset_type='train'):
        super().__init__()

        dataset_path = config['path']
        
        self.data_names, self.tri_vertices, self.tri_faces, self.tri_vnormals, self.tri_fnormals, self.tri_cd1, self.tri_cd2 = load_mesh_data(dataset_path, dataset_type)
        _, self.streams_data, self.streams_fid = load_streams_data(dataset_path, dataset_type)

        self.streams_num = [len(data) for data in self.streams_data]
        self.streams_vec_data = [[points[1:, :] - points[0:-1, :] for points in data] for data in self.streams_data]  # get the segment vector for each streams 

        self.pv = get_projected_vectors(self.tri_vertices, self.streams_data)  # get the streams projected vectors


    def __getitem__(self, id):

        data = dict()
        data['name'] = self.data_names[id]

        data['v'] = self.tri_vertices[id]
        data['vn'] = self.tri_vnormals[id]
        data['f'] = self.tri_faces[id]
        data['fn'] = self.tri_fnormals[id]
        data['cd1'] = self.tri_cd1[id]
        data['cd2'] = self.tri_cd2[id]
        
        #  projected vectors pv (v to streams points)
        data['pv'] = self.pv[id]

        data['s_num'] = self.streams_num[id]
        data['fid'] = self.streams_fid[id]
        data['s_vec'] = self.streams_vec_data[id]

        # normal direction toward the z-axis
        for i in range(data['vn'].shape[0]):
            if data['vn'][i, 2] < 0:
                data['vn'][i, :] = - data['vn'][i, :]

        for i in range(data['fn'].shape[0]):
            if data['fn'][i, 2] < 0:
                data['fn'][i, :] = - data['fn'][i, :]

        return (data)
    

    def __len__(self):

        return (len(self.data_names))


if __name__ == '__main__':

    dataset_path = 'E:/SplineMesh/Datasets/LearningCDF'
    
    dataset = TriMeshDataset(dataset_path, dataset_type='test')

    print(len(dataset))

    print(dataset[0]['vn'])
    print(dataset[0]['fn'])

    data_loader = DataLoader(dataset, num_workers=4, batch_size=2, shuffle=False, drop_last=True, collate_fn=collate_fn)

    for batch, data in enumerate(data_loader):

        print(data['v'])
        # print(data['v_mark'].size())
        # print(data['v_mark'])


