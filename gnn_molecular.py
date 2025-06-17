import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# - The train and test datasets were pre-processed graphs. The train dataset contains 20,000 graphs, while the test dataset contains 2,000 graphs.
# - Each graph contains the following components:
# 
#     - `x`, the matrix containing node features, `[num_of_nodes, num_node_features=11]`
#     - `edge_index`, the matrix containing connection information about different nodes, `[2, num_of_edges]`
#     - `y`, the label for the graph, `scaler`. The value is set to `0` in the test dataset
#     - `pos`, the matrix containing the node positions, `[num_of_nodes, 3]`
#     - `edge_attr`, the matrix containing the edge information, `[num_edges, 4]`
#     - `names`, index for the graph. For example, `gdb_59377`
# 
# - Depending on the graph convolutional layer that is used, different components are needed. For the most basic application, `x`, `edge_index` and `y` will be used.

class QM_Dataset(Dataset):
    def __init__(self, path):
        super().__init__(root=".")
        self.data = torch.load(path)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

train_path = "train.pt"
test_path = "test.pt"

train_data_ = QM_Dataset(train_path)

# train dataset can be split for validation purposes
train_data, validate_data = torch.utils.data.random_split(train_data_, [19000, 1000])
test_data = QM_Dataset(test_path)

class DenseGNN(torch.nn.Module):
    def __init__(self, num_features=11, hidden_dim=256, output_dim=1, num_layers=6):
        super().__init__()
        
        # Input transformation
        self.initial_lin = Linear(num_features, hidden_dim)
        
        # Dense GCN layers with residual connections
        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(BatchNorm(hidden_dim))
        
        # Hierarchical residual blocks
        self.res_lin = Linear(hidden_dim * num_layers, hidden_dim)
        
        # Graph-level regression output
        self.output = Sequential(
            Linear(hidden_dim, hidden_dim//2),
            ReLU(),
            Linear(hidden_dim//2, output_dim)
        )
        
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial projection
        x = F.relu(self.initial_lin(x))
        layer_outputs = []
        
        # Dense message passing
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x_out = conv(x, edge_index)
            x_out = bn(x_out)
            x_out = F.relu(x_out)
            x_out = self.dropout(x_out)
            x = x + x_out  # Residual connection
            layer_outputs.append(x)
        
        # Concatenate layer outputs
        x = torch.cat(layer_outputs, dim=-1)
        x = self.res_lin(x)
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Final regression output
        return self.output(x).view(-1)


def train(loader):
    model.train()
    total_loss = 0
    criterion = torch.nn.L1Loss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs  # Weight by graphs in batch
    return total_loss / len(loader.dataset)  # True average over ALL training graphs

def eval(loader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            total_loss += criterion(out, batch.y).item() * batch.num_graphs
    return total_loss / len(loader.dataset)  # Normalized by total test graphs


# load the datasets
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
validate_loader = DataLoader(validate_data, batch_size=128)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DenseGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00125)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',    # Monitor validation loss
    patience=6,    # Wait 5 epochs before reducing LR
    factor=0.5,    # Multiply LR by factor when triggered
    verbose=True   # Print update messages
)


num_epochs = 100
best_val_loss = float('inf')

# For plotting
train_losses = []
val_losses = []

for epoch in range(1, num_epochs+1):
    train_loss = train(train_loader)
    val_loss = eval(validate_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Prediction function
def predict(model, test_loader, test_data):
    model.eval()
    y_pred = []
    
    # Pre-collect all names from the test dataset
    test_names = [test_data[i].name for i in range(len(test_data))]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            pred = model(batch)
            y_pred.extend(pred.cpu().numpy().flatten().tolist())
    
    # Create DataFrame with pre-collected names
    df = pd.DataFrame({
        "Idx": test_names[:len(y_pred)],  # Ensure equal length
        "labels": y_pred
    })
    
    return df


epochs = np.arange(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)

# Generate predictions
model.load_state_dict(torch.load('best_model.pth'))
df = predict(model, test_loader, test_data)

# Create submission DataFrame
#df = pd.DataFrame({"Idx": Idx, "labels": y_pred})
df.to_csv("submission.csv", index=False)

# upload solution
#df.columns = ['Idx', 'labels']
#df.to_csv("/data/submission1.csv", index=False)

