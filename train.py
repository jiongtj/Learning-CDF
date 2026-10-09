import os 
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TriMeshDataset, collate_fn

from model import SplineNet
from loss import get_total_loss
from utils.config import get_config, set_logger

import time
import wandb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_args():

    parser = argparse.ArgumentParser(description='Learning CDF')
    parser.add_argument('--config', '-c', type=str, default='./Configs/config_train.yaml', help='Config File Path')
    args = parser.parse_args()

    return (args)


def train(data_loader, model, loss_fn, optimizer, config, epoch):

    logger.info('*************************')
    logger.info('Begin Training at Epoch: %d.' %(epoch + 1))
    logger.info('*************************')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.train()

    for batch, data in enumerate(data_loader):

        v = data['v'].to(device)       # [batch_size, num_points, 3]
        vn = data['vn'].to(device)     # [batch_size, num_points, 3]
        pv = data['pv'].to(device)     # [batch_size, num_points, 3]

        f = data['f'].to(device)       # [batch_size, num_faces, 3]
        fn = data['fn'].to(device)     # [batch_size, num_faces, 3]
        cd1 = data['cd1'].to(device)   # [batch_size, num_faces, 3]
        cd2 = data['cd2'].to(device)   # [batch_size, num_faces, 3]

        fid = data['fid'].to(device)      # [batch_size, num_max_fid]
        s_vec = data['s_vec'].to(device)  # [batch_size, num_max_fid, 3]
        p_cd1, p_cd2 = model(torch.cat((v, vn, pv), dim=2), f)  # [batch_size, num_faces, 3]

        direction_loss, normal_direction_loss, direction_smooth_loss, streams_loss = loss_fn(p_cd1, p_cd2, v, fn, cd1, cd2, f, fid, s_vec)

        loss = config['loss_weights']['direction_loss_weight'] * direction_loss + \
            config['loss_weights']['normal_direction_loss_weight'] * normal_direction_loss  + \
            config['loss_weights']['direction_smooth_loss_weight'] * direction_smooth_loss  + \
            config['loss_weights']['streams_loss_weight'] * streams_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(batch + 1) % config['trainer_params']['logger_steps'] == 0:

            logger.info('epoch: %d, batch: %d, loss: %f, direction loss: %f, normal direction loss: %f, direction smooth loss: %f, streams loss: %f.' 
                        %(epoch + 1, batch + 1, loss.item(), direction_loss.item(), normal_direction_loss.item(), direction_smooth_loss.item(), streams_loss.item()))

    logger.info('*************************')
    logger.info('End Training at Epoch: %d.' %(epoch + 1))
    logger.info('*************************')

    return (0)


def valid(data_loader, model, loss_fn, config, epoch):
    
    logger.info('*************************')
    logger.info('Begin Validation at Epoch: %d.' %(epoch + 1))
    logger.info('*************************')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    valid_loss = 0.0
    valid_loss1 = 0.0
    valid_loss2 = 0.0
    valid_loss3 = 0.0
    valid_loss4 = 0.0

    model.eval()
    with torch.no_grad():

        for batch, data in enumerate(data_loader):

            v = data['v'].to(device)       # [batch_size, num_points, 3]
            vn = data['vn'].to(device)     # [batch_size, num_points, 3]
            pv = data['pv'].to(device)     # [batch_size, num_points, 3]
            
            f = data['f'].to(device)       # [batch_size, num_faces, 3]
            fn = data['fn'].to(device)     # [batch_size, num_faces, 3]
            cd1 = data['cd1'].to(device)   # [batch_size, num_faces, 3]
            cd2 = data['cd2'].to(device)   # [batch_size, num_faces, 3]

            fid = data['fid'].to(device)      # [batch_size, num_max_fid]
            s_vec = data['s_vec'].to(device)  # [batch_size, num_max_fid, 3]

            p_cd1, p_cd2 = model(torch.cat((v, vn, pv), dim=2), f)  # [batch_size, num_faces, 3]

            valid_direction_loss, valid_normal_direction_loss, valid_direction_smooth_loss, valid_streams_loss = loss_fn(p_cd1, p_cd2, v, fn, cd1, cd2, f, fid, s_vec)

            valid_loss += config['loss_weights']['direction_loss_weight'] * valid_direction_loss.item() + \
                config['loss_weights']['normal_direction_loss_weight'] * valid_normal_direction_loss.item()  + \
                config['loss_weights']['direction_smooth_loss_weight'] * valid_direction_smooth_loss.item()  + \
                config['loss_weights']['streams_loss_weight'] * valid_streams_loss.item() 

            valid_loss1 += valid_direction_loss.item()
            valid_loss2 += valid_normal_direction_loss.item()
            valid_loss3 += valid_direction_smooth_loss.item()
            valid_loss4 += valid_streams_loss.item()

        valid_loss /= len(data_loader)
        valid_loss1 /= len(data_loader)
        valid_loss2 /= len(data_loader)
        valid_loss3 /= len(data_loader)
        valid_loss4 /= len(data_loader)
    
    logger.info('epoch: %d, valid loss: %f, valid direction loss: %f, valid normal direction loss: %f, valid direction smooth loss: %f, valid streams loss: %f.'
                %(epoch + 1, valid_loss, valid_loss1, valid_loss2, valid_loss3, valid_loss4))

    logger.info('***************************')
    logger.info('End validation at epoch: %d.' %(epoch + 1))
    logger.info('***************************')    

    return (0)


def run(config):

    logger.info('Learning CDF')

    # create training dataset and dataloader
    training_dataset = TriMeshDataset(config['dataset_params'], dataset_type='valid')
    training_dataloader = DataLoader(training_dataset, num_workers=4, batch_size=config['trainer_params']['batch_size'], shuffle=True, drop_last=True, collate_fn=collate_fn)

    # create validation dataset and dataloader
    validation_dataset = TriMeshDataset(config['dataset_params'], dataset_type='valid')
    validation_dataloader = DataLoader(validation_dataset, num_workers=4, batch_size=config['trainer_params']['batch_size'], shuffle=False, drop_last=True, collate_fn=collate_fn)

    logger.info('Size of training dataset: %d.' %(len(training_dataset)))
    logger.info('Size of training dataloader: %d.' %(len(training_dataloader)))
    logger.info('Size of validation dataset: %d.' %(len(validation_dataset)))
    logger.info('Size of validation dataloader: %d.' %(len(validation_dataloader)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set model, loss, optimizer, scheduler
    model = SplineNet(config).to(device)
    loss_fn = get_total_loss
    optimizer = optim.Adam(model.parameters(), lr=config['trainer_params']['learning_rate'])
    

    # training loop 
    for epoch in range(0, config['trainer_params']['epochs']):

        train(training_dataloader, model, loss_fn, optimizer, config, epoch)
        valid(validation_dataloader, model, loss_fn, config, epoch)

        # save model 
        torch.save(model.state_dict(), os.path.join(config['exp_params']['path'], f'model_{epoch + 1}.pth'))


if __name__ == '__main__':

    args = set_args()
    config = get_config(args.config)
    logger = set_logger(config['logger_params'])

    setup_seed(20)
    run(config)




