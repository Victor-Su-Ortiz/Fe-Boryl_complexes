import os
import torch
import re
import numpy as np
from scipy.spatial.distance import pdist, squareform

def read_xyz(file_path):
    """
    Reads an XYZ file and extracts the atom types and coordinates.

    Args:
        file_path (str): The path to the XYZ file.

    Returns:
        tuple: A tuple containing the atom types (list) and coordinates (numpy array).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    atom_count = int(lines[0].strip())
    comment = lines[1].strip()
    
    atom_types = []
    coordinates = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atom_types.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return atom_types, np.array(coordinates)



def create_graph(coordinates, threshold=1.5):
    """
    Creates a graph based on the coordinates of atoms.

    Args:
        coordinates (numpy array): The coordinates of atoms.
        threshold (float, optional): The distance threshold for creating edges. Defaults to 1.5.

    Returns:
        torch.Tensor: The edge index of the graph.
    """
    num_atoms = coordinates.shape[0]
    edge_index = []
    
    # Calculate pairwise distances
    distances = squareform(pdist(coordinates))
    
    # Create edges based on the distance threshold
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distances[i, j] <= threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# Assuming `data` is the Data object created previously
def save_graph_data(data, file_path):
    torch.save(data, file_path)
    print(f"Graph data saved to {file_path}")

# Example usage
# save_path = './data/torch_processed/graph_data.pth'
# save_graph_data(edge_index, save_path)


# Directory containing the XYZ files
directory_path = '/Users/victorsu-ortiz/Desktop/Fe-Boryl_complexes/data/xyz_molsimp'

# List all files in the directory
file_names = [f for f in os.listdir(directory_path) if f.endswith('.xyz')]

# Read each file and store the data
data = []
edge_indices = []

for idx, file_name in enumerate(file_names):
    file_path = os.path.join(directory_path, file_name)
    atom_types, coordinates = read_xyz(file_path)
    edge_index = create_graph(coordinates)
    # Define the regex pattern to extract the number before the file extension
    pattern = r'_(\d+)\.'

    # Search for the pattern in the file name
    match = re.search(pattern, file_name)

    # Extract the number if found
    number = match.group(1)
    edge_indices.append({"id": number, "edge_index": edge_index})
    save_path = f'./data/torch_processed/{file_name.split(".")[0]}.pth'
    save_graph_data(edge_indices[idx], save_path)

    tensor = torch.tensor(coordinates)
    data.append((file_name, atom_types, tensor))

# Example of accessing the data
for file_name, atom_types, tensor in data:
    print(f'File: {file_name}')
    print(f'Atom types: {atom_types}')
    print(f'Coordinates tensor: {tensor}')
    print()

