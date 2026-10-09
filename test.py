import os
import numpy as np 
import argparse
import torch 
from torch.utils.data import DataLoader
from model import SplineNet
from dataset import TriMeshDataset,collate_fn
from loss import get_total_loss
from utils.config import get_config, set_logger

import igl


def set_args():
    
    parser = argparse.ArgumentParser(description='LearningCDF')
    parser.add_argument('--config', '-c', type=str, default='./Configs/config_test.yaml', help='Config file path')
    args = parser.parse_args()

    return(args)


def write_CDF(fn, p_cd1, p_cd2, name):

    output_path = config['output_params']['path']
    fn = fn.squeeze(dim=0).cpu().numpy()
    p_cd1 = p_cd1.squeeze(dim=0).cpu().numpy()
    p_cd2 = p_cd2.squeeze(dim=0).cpu().numpy()
    
    geometry = np.concatenate((fn, p_cd1, p_cd2), axis=1)
    np.savetxt(os.path.join(output_path, str(name[0]) + '_CDF_2.1.csv'), geometry, fmt='%f', delimiter=',')


def test(data_loader, model, loss_fn, config):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    average_direction_loss = 0.0
    average_normal_direction_loss = 0.0
    average_direction_smooth_loss = 0.0
    average_streams_loss = 0.0

    model.eval()
    with torch.no_grad():

        for batch, data in enumerate(data_loader):
            
            v = data['v'].to(device)
            vn = data['vn'].to(device)
            pv = data['pv'].to(device)

            f = data['f'].to(device)
            fn = data['fn'].to(device)
            cd1 = data['cd1'].to(device)
            cd2 = data['cd2'].to(device)

            fid = data['fid'].to(device)      # [batch_size, num_max_fid]
            s_vec = data['s_vec'].to(device)  # [batch_size, num_max_fid, 3]
            name = data['names']

            p_cd1, p_cd2 = model(torch.cat((v, vn, pv), dim=2), f) # [batch_size, num_faces, 3]

            direction_loss, normal_direction_loss, direction_smooth_loss, streams_loss = loss_fn(p_cd1, p_cd2, v, fn, cd1, cd2, f, fid, s_vec)
            
            loss = direction_loss + normal_direction_loss + direction_smooth_loss + streams_loss

            print('mesh id: %d, loss: %f, direction loss: %f,  normal direction loss: %f, direction smooth loss: %f, streams_loss: %f.'
                  %(batch, loss.item(), direction_loss.item(), normal_direction_loss.item(), direction_smooth_loss.item(), streams_loss.item()))

            write_CDF(fn, p_cd1, p_cd2, name)
        
            average_direction_loss += direction_loss.item()
            average_normal_direction_loss += normal_direction_loss.item()
            average_direction_smooth_loss += direction_smooth_loss.item()
            average_streams_loss += streams_loss.item()

    average_direction_loss /= len(data_loader)
    average_normal_direction_loss /= len(data_loader)
    average_direction_smooth_loss /= len(data_loader)
    average_streams_loss /= len(data_loader)

    print('average direction loss: %f, average normal direction loss: %f, average direction smooth loss: %f, average streams loss: %f.' 
            %(average_direction_loss, average_normal_direction_loss, average_direction_smooth_loss, average_streams_loss))


def run(config):

    logger.info('SplineMesh Process')

    # create test dataset and dataloader
    test_dataset = TriMeshDataset(config['dataset_params'], dataset_type='test')
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn)

    logger.info('Size of testing dataset: %d.', len(test_dataset))
    logger.info('SIze of testing dataloader: %d.', len(test_dataloader))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model 
    model = SplineNet(config).to(device)
    model_path = os.path.join(config['exp_params']['path'], config['exp_params']['name'])
    model.load_state_dict(torch.load(model_path))

    loss_fn = get_total_loss
    
    logger.info('***************************')
    logger.info('Begin testing')
    logger.info('***************************')

    test(test_dataloader, model, loss_fn, config)

    logger.info('***************************')
    logger.info('End testing')
    logger.info('***************************')


if __name__ == '__main__':
    
    args = set_args()
    config = get_config(args.config)
    logger = set_logger(config['logger_params'])

    run(config)

