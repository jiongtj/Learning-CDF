import numpy as np
import scipy 


def get_nearest_points(mesh_points, streams_points):

    streams_points = np.concatenate(streams_points, axis=0)
    distance = scipy.spatial.distance.cdist(mesh_points, streams_points, 'euclidean')  # [num_mesh_points, num_stream_points]
    
    min_index = np.argmin(distance, axis=1)
    
    pv = streams_points[min_index, :] - mesh_points  # [num_mesh_points, 3]

    return (pv)

# get all the projected vectors from mesh poitns to stream points
def get_projected_vectors(total_tri_vertices, total_streams_data):

    total_pv = list()

    for i in range(len(total_tri_vertices)):

        pv = get_nearest_points(total_tri_vertices[i], total_streams_data[i])
        total_pv.append(pv)

    return (total_pv)
