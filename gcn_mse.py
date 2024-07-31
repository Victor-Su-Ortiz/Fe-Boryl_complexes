import os
import torch
import csv
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torch_geometric.nn import GraphConv, global_mean_pool
from torcheval.metrics import R2Score
from matplotlib import pyplot as plt


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

class GCNNet_mse(torch.nn.Module):
    def __init__(self, num_node_features, hp):
        super(GCNNet_mse, self).__init__()

        self.use_batch_norm = hp['use_batch_norm']
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if self.use_batch_norm else None
        self.metric = R2Score()
        hidden_dims = [hp[f'hidden_dim{i+1}'] for i in range(6)]
        num_layers = [hp[f'num_layers{i+1}'] for i in range(6)]

        # Define the Graph Convolution layers
        for i in range(len(hidden_dims)):
            for j in range(num_layers[i]):
                if i == 0 and j == 0:
                    in_features = num_node_features
                elif j == 0:
                    in_features = hidden_dims[i-1]
                else:
                    in_features = hidden_dims[i]
                out_features = hidden_dims[i]
                self.convs.append(GraphConv(in_features, out_features))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(out_features))

        # Define the dense layer
        self.dense = torch.nn.Linear(hidden_dims[-1], hidden_dims[-1])

        # Define the output layer
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph Convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

        # Aggregate node features into a single graph-level representation
        x = global_mean_pool(x, data.batch)

        # Dense layer
        x = self.dense(x)
        x = torch.nn.functional.relu(x)

        # Output layer
        mu = self.fc_mu(x).squeeze(-1)

        return mu

# Function to calculate mean and std of 'y'
def calculate_mean_std(loader):
    y_values = []
    for batch in loader:
        y_values.append(batch.y)
    y_values = torch.cat(y_values, dim=0)
    mean = y_values.mean(dim=0)
    std = y_values.std(dim=0)
    return mean, std

# Create the loss function
# loss_fn = torch.nn.GaussianNLLLoss()
loss_fn = torch.nn.MSELoss()

torch.manual_seed(2024)

def load_all_graphs(folder_path):
    # Get a list of all .pt files in the folder
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pth')]

    # Load all the graphs from the files
    all_graphs = []
    for file_path in file_paths:
        graph_dict = torch.load(file_path)
        all_graphs.append(graph_dict)

    return all_graphs

all_graphs = load_all_graphs('/Users/victorsu-ortiz/Desktop/Fe-Boryl_complexes/data/torch_processed')
for graph in all_graphs:
    HS_E_red = torch.tensor(graph['HS_E_red'].reshape(-1, 1))
    LS_E_red = torch.tensor(graph['LS_E_red'].reshape(-1, 1))
    graph.y = torch.cat((HS_E_red, LS_E_red), dim=1)

# Define the number of features for the nodes
num_node_features = 4

def run_gcn_mse(hp):

    # Retrieve job_id from hyperparameters, if available
    job_id = hp.get('job_id', 'unknown')

    # Mix up the seed for different evals
    if job_id == 'unknown':
        seed = 2024
    else:
        seed = int(job_id) + 2024

    torch.manual_seed(seed)

    data_loader = DataLoader(all_graphs, batch_size=hp['batch_size'], shuffle=True)

    # Calculate the number of samples for each dataset
    num_graphs = len(all_graphs)
    num_train = int(num_graphs * 0.7)
    num_val = int(num_graphs * 0.15)
    num_test = num_graphs - num_train - num_val  # Ensure the remaining data goes to the test set

    # Split the dataset
    train_data, val_data, test_data = random_split(data_loader.dataset, [num_train, num_val, num_test])

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=hp['batch_size'], shuffle=False)

    mean, std = calculate_mean_std(train_loader)

    # Create the model and move it to the device
    model = GCNNet_mse(
        num_node_features,
        hp
    ).to(device)


    # Create a list to store validation losses
    val_losses = []

    # Create a list to store R2 score
    r2Score = []

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'])

    # Create the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=hp['patience'], factor=hp['scheduler_factor'], min_lr=1e-6)

    # Training loop
    for epoch in range(hp['epochs']):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            if(hp['z-normalize']):
                actual_output = (batch.y.float()-mean)/std
                loss = loss_fn(output,actual_output.float())
            else:
                loss = loss_fn(output, batch.y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        actual_outputs = []
        predictions = []
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                if(hp['z-normalize']):
                    actual_output = (batch.y.float()-mean)/std
                    val_loss += loss_fn(output,actual_output.float()).item()
                    predictions.append(output)
                    actual_outputs.append(actual_output)
                else:
                    val_loss += loss_fn(output, batch.y.float()).item()
                    predictions.append(output)
                    actual_outputs.append(batch.y.float())
            val_loss /= len(val_loader)
        
        
        
        val_outputs = torch.cat(actual_outputs)
        val_predictions = torch.cat(predictions)
        model.metric.update(val_predictions, val_outputs)
        r2score = model.metric.compute().item()
        
        print(f'Epoch {epoch+1} of job {job_id}, Validation Loss: {val_loss:.7f}')
        print(f'Epoch {epoch+1} of job {job_id}, R2 Score: {r2score:.7f}')

        # Step the scheduler
        scheduler.step(val_loss)

        # Append the epoch and validation loss to the list
        val_losses.append([epoch+1, val_loss])
        r2Score.append([epoch+1, r2score])

    # Save validation losses to a CSV file
    os.makedirs(f'./val_losses/gcn_noxyz', exist_ok=True)
    with open(f'./val_losses/gcn_noxyz/val_losses_{job_id}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Validation Loss'])
        writer.writerows(val_losses)

    os.makedirs(f'./r2_score/gcn_noxyz', exist_ok=True)
    with open(f'./r2_score/gcn_noxyz/r2_score_{job_id}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'R2 Score'])
        writer.writerows(r2Score)
    
    # Plotting predicted vs actual values
    # plt.figure(figsize=(8, 6))
    # plt.scatter(val_outputs.numpy(), val_predictions.numpy(), alpha=0.5)
    # plt.plot([val_outputs.min(), val_outputs.max()], [val_outputs.min(), val_outputs.max()], 'r', lw=2)
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Predicted vs Actual Values')
    # plt.show()

    # # Save the model
    # model_save_dir = 'saved_retrained_ensemble_exp12_gcn_mse_noxyz'
    # os.makedirs(model_save_dir, exist_ok=True)
    # model_save_path = os.path.join(model_save_dir, f'model_{job_id}.pth')
    # torch.save(model.state_dict(), model_save_path)

    # Return the negative final validation loss as the metric for hyperparameter tuning
    return -val_loss, r2score

    # return model

if __name__ == "__main__":
    hp = {
        'batch_size': 16,
        'patience': 10,
        'scheduler_factor': 0.8,
        'hidden_dim1': 32,
        'hidden_dim2': 64,
        'hidden_dim3': 128,
        'hidden_dim4': 64,
        'hidden_dim5': 32,
        'hidden_dim6': 16,
        'num_layers1': 2,
        'num_layers2': 2,
        'num_layers3': 2,
        'num_layers4': 2,
        'num_layers5': 2,
        'num_layers6': 2,
        'lr': 0.00005,
        'epochs': 200,
        'use_batch_norm': True,
        'z-normalize': True,
    }

    run_gcn_mse(hp)



