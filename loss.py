import torch
import torch.nn.functional as F


# input: x [batchs_size, num_points, 3]
#        k number of neighbourings
# output: idx [batch_size, num_points, k] 
def get_knn_idx(x, k=10):

    inner = 2 * torch.matmul(x, x.transpose(2, 1))  # [batch_size, num_points, num_points]
    xx = torch.sum(x ** 2, dim=2, keepdim=True)     # [batchs_size, num_points, 1]
    pairwise_distance = xx - inner  +  xx.transpose(2, 1) # [batch_size, num_points, num_points]

    idx = torch.topk(pairwise_distance, k=k+1, largest=False, dim=2)[1]  # [batch_size, num_points, K]
    idx = idx[:, :, 1:]

    return (idx)


# get neighbouring features
# input: x  [batch_size, num_points, num_dims]
#       idx [batch_size, num_points, K]
# output: x [batch_size, num_points, K, num_dims]
def get_neighbouring_feature(x, idx):

    batch_size, num_points, num_dims = x.size()
    k = idx.size(2)
    
    # (1) use index selection
    # idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    # idx = idx + idx_base
    # y = x.view(batch_size * num_points, num_dims)[idx.view(-1), :]
    # y = y.view(batch_size, num_points, k, num_dims)

    # (2) use torch.gather function
    # y = x.clone()
    x = torch.gather(x, 1, idx.reshape(batch_size, num_points * k).unsqueeze(dim=-1).expand(-1, -1, num_dims))
    x = x.view(batch_size, num_points, k, num_dims)

    # x = torch.gather(x.unsqueeze(dim=2).expand(-1, -1, k, -1), 1, idx.unsqueeze(dim=3).expand(-1, -1, -1, num_dims))  # [batch_size, num_points, k, num_dims]

    return (x) 

# input: x [batch_size, num_faces/num_points, 3]
def normalization(x):

    l2_norm = torch.linalg.vector_norm(x, dim=2, keepdim=True)  # [batch_size, num_points, 1]
    l2_norm = torch.clamp(l2_norm, min=1e-8)   # set small value for numerical stability
    x = x / l2_norm

    return (x)

# f: direction fields [batch_size, num_points, k, 3]
# e: edge vectors [batch_size, num_points, k, 3]
# n: normals [batch_size, num_points, k, 3]
def get_local_frame_coordinates(f, e, n):

    batch_size, num_points, k, _ = f.size()

    # project edge vectors onto tangent plane
    inner_value = torch.linalg.vecdot(e, n, dim=3).view(batch_size, num_points, k, 1)
    frame_x = e - inner_value * n
    frame_x = frame_x / torch.clamp(torch.linalg.vector_norm(frame_x, dim=3, keepdim=True), min=1e-8)  # [batch_size, num_points, k, 3]
    frame_y = torch.linalg.cross(n, frame_x)                                                           # [batch_size, num_points, k, 3]

    # get local coordinates for direction field on local frame (frame_x, frame_y, expand_n)
    coord_x = torch.linalg.vecdot(f, frame_x, dim=3).view(batch_size, num_points, k, 1)  # [batch_size, num_points, k, 1]
    coord_y = torch.linalg.vecdot(f, frame_y, dim=3).view(batch_size, num_points, k, 1)  # [batch_size, num_points, k, 1]
    coord = torch.cat((coord_x, coord_y), dim=3)  # [batch_size, num_points, k, 2]

    # normalization
    coord = coord / torch.clamp(torch.linalg.vector_norm(coord, dim=3, keepdim=True), min=1e-8)  # [batch_size, num_points, k, 2]

    return (coord)


def get_direction_loss(p_cd1, p_cd2, cd1, cd2, fn):

    # normalization
    cd1 = normalization(cd1)
    cd2 = normalization(cd2)

    values_1 = torch.linalg.vecdot(p_cd1, torch.linalg.cross(fn, cd1), dim=2)  # [batch_size, num_faces]
    values_1 = values_1 ** 2 

    values_2 = torch.linalg.vecdot(p_cd2, torch.linalg.cross(fn, cd2), dim=2)  # [batch_size, num_faces]
    values_2 = values_2 ** 2

    # swap orders
    values_3 = torch.linalg.vecdot(p_cd1, torch.linalg.cross(fn, cd2), dim=2)  # [batch_size, num_points]
    values_3 = values_3 ** 2

    values_4 = torch.linalg.vecdot(p_cd2, torch.linalg.cross(fn, cd1), dim=2)  # [batch_size num_points]
    values_4 = values_4 ** 2

    loss = torch.minimum(values_1 + values_2, values_3 + values_4) # [batch_size, num_points]
    loss = torch.mean(loss)

    return (loss)


def get_normal_direction_loss(p_cd1, p_cd2, fn):

    value_1 = torch.sum(p_cd1 * fn, dim=2)  # [batch_size, num_faces]
    value_2 = torch.sum(p_cd2 * fn, dim=2)  # [batch_size, num_faces]

    loss = torch.mean(value_1 ** 2 + value_2 ** 2)

    return (loss)


def get_direction_smooth_loss(p_cd1, p_cd2, fn, v, faces, k=10):

    batch_size, num_faces, _ = p_cd1.size()

    # get barycenter
    barycenter = torch.gather(v, 1, faces.view(batch_size, num_faces * 3).unsqueeze(dim=-1).expand(-1, -1, 3)) # [batch_size, num_faces * 3, 3]
    barycenter = barycenter.view(batch_size, num_faces, 3, 3)
    barycenter = torch.mean(barycenter, dim=2)  # [batch_size, num_faces, 3]

    knn_idx = get_knn_idx(barycenter, k=10)  # [batch_size, num_faces, k]

    nn_d1 = get_neighbouring_feature(p_cd1, knn_idx)  # [batch_size, num_faces, k, 3]
    nn_d2 = get_neighbouring_feature(p_cd2, knn_idx)  # [batch_size, num_faces, k, 3]
    nn_n = get_neighbouring_feature(fn, knn_idx)  # [batch_size, num_faces, k, 3]
    nn_v = get_neighbouring_feature(barycenter, knn_idx)  # [batch_size, num_faces, k, 3]

    expand_d1 = p_cd1.unsqueeze(dim=2).expand(-1, -1, k, -1)  # [batch_size, num_faces, k, 3]
    expand_d2 = p_cd2.unsqueeze(dim=2).expand(-1, -1, k, -1)  # [batch_size, num_faces, k, 3]
    expand_n = fn.unsqueeze(dim=2).expand(-1, -1, k, -1)  # [batch_size, num_faces, k, 3]
    expand_v = barycenter.unsqueeze(dim=2).expand(-1, -1, k, -1)  # [batch_size, num_faces, k, 3]

    edges = nn_v - expand_v

    coord_1 = get_local_frame_coordinates(expand_d1, edges, expand_n)  # [batch_size, num_faces, k, 2]
    coord_2 = get_local_frame_coordinates(nn_d1, edges, nn_n)          # [batch_size, num_faces, k, 2]

    coord_3 = get_local_frame_coordinates(expand_d2, edges, expand_n)  # [batch_size, num_faces, k, 2]
    coord_4 = get_local_frame_coordinates(nn_d2, edges, nn_n)          # [batch_size, num_faces, k, 2]
    
    # compute loss
    # rotata 90 degree for coord_j and compute theta
    # loss = min(1.0 - cos(theta) ^2, 1.0 - cos(pi/2 + theta)^2) = min(sin(theta)^2, cos(theta)^2)

    # rotate 90 degree
    coord_2 = coord_2[:, :, :, [1, 0]]
    coord_2[:, :, :, 0] = - coord_2[:, :, :, 0]

    coord_4 = coord_4[:, :, :, [1, 0]] 
    coord_4[:, :, :, 0] = - coord_4[:, :, :, 0]

    values_1 = torch.linalg.vecdot(coord_1, coord_2, dim=3)  # [batch_size, num_faces, k]
    values_1 = values_1 ** 2

    values_2 = torch.linalg.vecdot(coord_3, coord_4, dim=3)  # [batch_size, num_faces, k]
    values_2 = values_2 ** 2

    values_3 = torch.linalg.vecdot(coord_1, coord_4, dim=3)  # [batch_size, num_faces, k]
    values_3 = values_3 ** 2

    values_4 = torch.linalg.vecdot(coord_3, coord_2, dim=3)  # [batch_size, num_faces, k]
    values_4 = values_4 ** 2

    loss = torch.minimum(values_1 + values_2, values_3 + values_4) # [batch_size, num_faces, k]
    loss = torch.mean(loss)

    return (loss)


def get_streams_loss(p_cd1, p_cd2, fn, f_id, s_vec):

    batch_size, num_faces, _ = p_cd1.size()
    num_max_fid = f_id.size(1)

    f_id_mask = f_id > -1     # [batch_size, num_max_fid]
    
    f_id = f_id_mask * f_id   # [batch_size, num_max_fid]

    f_p_cd1 = torch.gather(p_cd1, 1, f_id.unsqueeze(dim=-1).expand(-1, -1, 3))  # [batch_size, num_max_fid, 3]
    f_p_cd2 = torch.gather(p_cd2, 1, f_id.unsqueeze(dim=-1).expand(-1, -1, 3))  # [batch_size, num_max_fid, 3]
    f_fn = torch.gather(fn, 1, f_id.unsqueeze(dim=-1).expand(-1, -1, 3))        # [batch_size, num_max_fid, 3]

    # normalization
    s_vec = normalization(s_vec)

    value_1 = torch.linalg.vecdot(f_p_cd1, torch.linalg.cross(f_fn, s_vec), dim=2)  # [batch_size, num_max_fid]
    value_1 = value_1 ** 2

    value_2 = torch.linalg.vecdot(f_p_cd2, torch.linalg.cross(f_fn, s_vec), dim=2)  # [batch_size, num_max_fid]
    value_2 = value_2 ** 2

    loss = torch.minimum(value_1, value_2)  # [batch_size, num_max_fid]

    loss = loss * f_id_mask  #[batch_size, num_max_fid]
    loss = torch.sum(loss, dim=1) / torch.sum(f_id_mask, dim=1)  # [batch_size]
    loss = torch.mean(loss)

    return (loss)


# input: p_cd1, p_cd2, fn, d1, d2 [batch_size, num_faces, 3],
#        v [batch_size, num_points, 3]
#        faces [num_faces, 3]

def get_total_loss(p_cd1, p_cd2, v, fn, cd1, cd2, faces, f_id, s_vec, k=10):

    direction_loss = get_direction_loss(p_cd1, p_cd2, cd1, cd2, fn)
    normal_direction_loss = get_normal_direction_loss(p_cd1, p_cd2, fn)
    direction_smooth_loss = get_direction_smooth_loss(p_cd1, p_cd2, fn, v, faces, k)
    streams_loss = get_streams_loss(p_cd1, p_cd2, fn, f_id, s_vec)

    return (direction_loss, normal_direction_loss, direction_smooth_loss, streams_loss)