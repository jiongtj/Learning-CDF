import os 
import pickle 

# load triangle mesh data with CDF
def load_mesh_data(data_path, data_type):

    file_name = os.path.join(data_path, f'TriMesh_{data_type}.pkl')

    with open(file_name, 'rb') as file:
        data = pickle.load(file)

        names = data['names']                # list (string names)   
        tri_vertices = data['tri_vertices']  # list (array tri vertex position)  
        tri_faces = data['tri_faces']        # list (array tri face index)

        tri_vnormals = data['tri_vnormals']  # list (array tri vertex normal)
        tri_fnormals = data['tri_fnormals']  # list (array tri face normal)

        tri_cd1 = data['tri_cd1']  # list (array conjugate direction u)
        tri_cd2 = data['tri_cd2']  # list (array conjugate direction v)

    return (names, tri_vertices, tri_faces, tri_vnormals, tri_fnormals, tri_cd1, tri_cd2)


# load streams data with face indices
def load_streams_data(data_path, data_type):

    file_name = os.path.join(data_path, f'Streams_{data_type}.pkl')

    with open(file_name, 'rb') as file:
        data = pickle.load(file)

        names = data['names']              # list (string names)  
        streams = data['streams']          # list (array stream data)
        streams_fid = data['streams_fid']  # list (array stream fid)

    return (names, streams, streams_fid)
