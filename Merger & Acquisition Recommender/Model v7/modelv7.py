import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_matrix
import pickle

# 1. Load Data
nodes_df = pd.read_excel('dummy_working.xlsx')
edges_df = pd.read_csv('dummy_edges.csv')

# --- FIX: DROP DUPLICATES IMMEDIATELY ---
# This ensures N = 14177 for both features and the graph
nodes_df = nodes_df.drop_duplicates(subset=['Id']).reset_index(drop=True)

num_cols = ['indegree', 'outdegree', 'employees_total', 'revenue_usd', 
            'corp_fam_members', 'mark_val_usd', 'it_spending']
scaler = StandardScaler()
nodes_df[num_cols] = scaler.fit_transform(nodes_df[num_cols])

sic_dummies = pd.get_dummies(nodes_df['sic_desc'], prefix='sic')
eff_dummies = pd.get_dummies(nodes_df['efficiency'], prefix='eff')

# Combine features and store column names for the interface
X_df = pd.concat([nodes_df[num_cols], sic_dummies, eff_dummies], axis=1).astype(float)
feature_cols = X_df.columns.tolist() 
X = torch.tensor(X_df.values, dtype=torch.float)

# Mapping node IDs to indices
node_to_idx = {name: i for i, name in enumerate(nodes_df['Id'])}
idx_to_node = {i: name for name, i in node_to_idx.items()}
num_nodes = len(node_to_idx)

# Filter edges to only include nodes present in the cleaned nodes list
valid_edges = edges_df[edges_df['source'].isin(node_to_idx) & 
                      edges_df['target'].isin(node_to_idx)]
edge_indices = [[node_to_idx[r['source']], node_to_idx[r['target']]] for _, r in valid_edges.iterrows()]
edge_index = torch.tensor(edge_indices, dtype=torch.long).t()

# 2. Define Model Structure
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        # Sparse Matrix Multiplication for GCN layer
        x = F.relu(self.lin1(torch.spmm(adj, x)))
        x = self.lin2(torch.spmm(adj, x))
        return x

def get_adj_norm(edge_index, num_nodes):
    row, col = edge_index.numpy()
    adj = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    adj = (adj + adj.T).tocoo()
    adj.setdiag(1)
    d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = coo_matrix((d_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))))
    adj_norm = d_mat.dot(adj).dot(d_mat).tocoo()
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(adj_norm.data.astype(np.float32)), [num_nodes, num_nodes])

# 3. Train the Model
adj = get_adj_norm(edge_index, num_nodes)
model = GCNLinkPredictor(X.shape[1], 32, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    z = model(X, adj)
    pos_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    neg_idx = torch.randint(0, num_nodes, (2, edge_index.shape[1]))
    neg_scores = (z[neg_idx[0]] * z[neg_idx[1]]).sum(dim=-1)
    loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean() - \
           torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
    loss.backward()
    optimizer.step()

# 4. Generate Model Files
torch.save(model.state_dict(), 'gnn_acquisition_model.pth')
metadata = {
    'node_to_idx': node_to_idx,
    'idx_to_node': idx_to_node,
    'num_cols': num_cols,
    'feature_cols': feature_cols,
    'scaler': scaler,
    'in_channels': X.shape[1],
    'hidden_channels': 32,
    'out_channels': 16
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved: gnn_acquisition_model.pth and model_metadata.pkl")

from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, X, adj, pos_edges, neg_edges):
    model.eval()
    with torch.no_grad():
        z = model(X, adj)

        pos_scores = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=-1)
        neg_scores = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=-1)

        scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        labels = np.hstack([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores))
        ])

        probs = torch.sigmoid(torch.tensor(scores)).numpy()

        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)

    return auc, ap

def split_edges(edge_index, val_ratio=0.1, test_ratio=0.1):
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)

    test_size = int(num_edges * test_ratio)
    val_size = int(num_edges * val_ratio)

    test_edges = edge_index[:, perm[:test_size]]
    val_edges = edge_index[:, perm[test_size:test_size+val_size]]
    train_edges = edge_index[:, perm[test_size+val_size:]]

    return train_edges, val_edges, test_edges

def negative_sampling(num_nodes, num_samples):
    return torch.randint(0, num_nodes, (2, num_samples))

train_e, val_e, test_e = split_edges(edge_index)

val_neg = negative_sampling(num_nodes, val_e.shape[1])
test_neg = negative_sampling(num_nodes, test_e.shape[1])

val_auc, val_ap = evaluate(model, X, adj, val_e, val_neg)
test_auc, test_ap = evaluate(model, X, adj, test_e, test_neg)

print(f"Validation AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
print(f"Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}")