import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [batch_size, num_points, k]
    return idx


def get_graph_feature(x, k=20, idx=None):

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)   # [batch_size, num_points, k]
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # [batch_size, num_points, num_dims]  -> [batch_size * num_points, num_dims] #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


# feature normalization
# input: x [batch_size, num_points, 3]
def normalization(x):

    l2_norm = torch.linalg.vector_norm(x, dim=2, keepdim=True)  # [batch_size, num_points, 1]
    l2_norm = torch.clamp(l2_norm, min=1e-8)   # set small value for numerical stability
    x = x / l2_norm

    return (x)


# input: x [batch_size, num_features, num_points];
#        faces [batch_size, num_faces, 3]
# output: x [batch_size, num_features, num_faces]
def point2face_features(x, faces):

    batch_size, num_faces, _ = faces.size()
    num_features = x.size(1)

    x = x.transpose(2, 1)  # [batch_size, num_points, num_features]
    
    x = torch.gather(x, 1, faces.view(batch_size, num_faces * 3).unsqueeze(dim=-1).expand(-1, -1, num_features))  # [batch_size, num_faces * 3, num_features]
    x = x.view(batch_size, num_faces, 3, num_features)

    x = torch.mean(x, dim=2)  # [batch_size, num_faces, num_features]
    x = x.transpose(2, 1)     # [batch_size, num_features, num_faces]

    return (x)


class PointFeatures(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.k = config['trainer_params']['k']
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=18, out_channels=64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.maxpool1d = nn.AdaptiveMaxPool1d(1) 
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)

        self.conv6 = nn.Sequential(nn.Linear(in_features=256, out_features=512, bias=False),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.dropout6 = nn.Dropout(p=0.1)

        self.conv7 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                  nn.BatchNorm1d(256),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.dropout7 = nn.Dropout(p=0.45)

        self.conv8 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=1, bias=False)


    def forward(self, x):

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = torch.max(x, dim=3, keepdim=False)[0]  # [batch_size, 64, num_points]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = torch.max(x, dim=3, keepdim=False)[0]  # [batch_sie, 64, num_points]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = torch.max(x, dim=3, keepdim=False)[0]  # [batch_size, 128, num_points]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = torch.max(x, dim=3, keepdim=False)[0]  # [batch_size, 256, num_points]

        # local feature defined on each points
        point_feature = torch.cat((x1, x2, x3, x4), dim=1)  # [batch_size, 512, num_points]
        
        # global feature defined on the point cloud
        x = self.conv5(point_feature)  # [batch_size, 128, num_points]
        x1 = self.maxpool1d(x).squeeze(dim=2)  # [batch_size, 128]
        x2 = self.avgpool1d(x).squeeze(dim=2)  # [batch_size, 128]
        x = torch.cat((x1, x2), dim=1)         # [batch_size, 256]

        x = self.conv6(x)  # [batch_size, 512]
        x = self.dropout6(x)

        x = self.conv7(x) # [batch_size, 256]
        x = self.dropout7(x) 

        # combination of local feature and global feature
        num_points = point_feature.size(2)
        x = x.unsqueeze(dim=2).expand(-1, -1, num_points)  # [batch_size, 256, num_points]
        x = torch.cat((point_feature, x), dim=1)  # [batch_size, 768, num_points]

        x = self.conv8(x)  # [batch_size, 256, num_points]

        return (x)


class FacePredictions(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU())
        
        self.mlp2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU())
        
        self.mlp3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=False))

        self.mlp4 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU())

        self.mlp5 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU())
        
        self.mlp6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1, bias=False))

      
    def forward(self, x, faces):
        
        x = point2face_features(x, faces)  # [batch_size, 256, num_faces]
        d1 = self.mlp1(x)   # [batch_size, 128, num_faces]
        d1 = self.mlp2(d1)  # [batch_size, 64, num_faces]
        d1 = self.mlp3(d1)  # [batch_size, 3, num_faces]

        d2 = self.mlp4(x)   # [batch_size, 128, num_faces]
        d2 = self.mlp5(d2)  # [batch_size, 64, num_faces]
        d2 = self.mlp6(d2)  # [batch_size, 3, num_faces]

        return (d1, d2)


class SplineNet(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.pointfeatures = PointFeatures(config)
        self.facepredictions = FacePredictions(config)

    
    def forward(self, x, faces):
        
        x = x.transpose(2, 1)                       # [batch_size, 6, num_points]
        x = self.pointfeatures(x)                   # [batch_size, 256, num_points]
        d1, d2 = self.facepredictions(x, faces)     # [batch_size, 3, num_faces]
        
        d1 = d1.transpose(2, 1)      # [batch_size, num_faces, 3]
        d2 = d2.transpose(2, 1)      # [batch_size, num_faces, 3]

        d1 = normalization(d1)       # [batch_size, num_faces, 3]
        d2 = normalization(d2)       # [batch_size, num_faces, 3]

        return (d1, d2)



    








