import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import pickle
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, adj):
        x = F.relu(self.lin1(torch.spmm(adj, x)))
        x = self.lin2(torch.spmm(adj, x))
        return x

class GNNInferenceInterface:
    def __init__(self, model_path='gnn_acquisition_model.pth', metadata_path='model_metadata.pkl'):
        with open(metadata_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.model = GCNLinkPredictor(self.meta['in_channels'], 32, 16)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.node_to_idx = self.meta['node_to_idx']
        self.idx_to_node = self.meta['idx_to_node']

    def prescribe(self, nodes_df, edges_df, top_n=10):
        # 1. Preprocess & Align
        nodes_df = nodes_df.drop_duplicates(subset=['Id']).reset_index(drop=True)
        nodes_df[self.meta['num_cols']] = self.meta['scaler'].transform(nodes_df[self.meta['num_cols']])
        
        sic_dummies = pd.get_dummies(nodes_df['sic_desc'], prefix='sic')
        eff_dummies = pd.get_dummies(nodes_df['efficiency'], prefix='eff')
        X_df = pd.concat([nodes_df[self.meta['num_cols']], sic_dummies, eff_dummies], axis=1)
        
        # Ensure all training features are present (zero-fill missing)
        for col in [c for c in self.meta['feature_cols'] if c not in X_df.columns]:
            X_df[col] = 0
        X_df = X_df[self.meta['feature_cols']].astype(float)
        
        X = torch.tensor(X_df.values, dtype=torch.float)
        num_nodes = len(self.node_to_idx)
        
        # 2. Build Adjacency Matrix
        valid_edges = edges_df[edges_df['source'].isin(self.node_to_idx) & edges_df['target'].isin(self.node_to_idx)]
        edge_indices = [[self.node_to_idx[r['source']], self.node_to_idx[r['target']]] for _, r in valid_edges.iterrows()]
        edge_idx = torch.tensor(edge_indices, dtype=torch.long).t()
        
        row, col = edge_idx.numpy()
        adj = coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
        adj = (adj + adj.T).tocoo()
        adj.setdiag(1)
        d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat = coo_matrix((d_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))))
        adj_norm = d_mat.dot(adj).dot(d_mat).tocoo()
        indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
        adj_tensor = torch.sparse_coo_tensor(indices, torch.from_numpy(adj_norm.data.astype(np.float32)), [num_nodes, num_nodes])

        # 3. Inference
        with torch.no_grad():
            z = self.model(X, adj_tensor)
            potential_pairs = torch.randint(0, num_nodes, (2, 50000))
            scores = torch.sigmoid((z[potential_pairs[0]] * z[potential_pairs[1]]).sum(dim=-1))
            top_scores, top_idx = torch.topk(scores, top_n)
            
            return [{"acquirer": self.idx_to_node[potential_pairs[0, i].item()], 
                     "target": self.idx_to_node[potential_pairs[1, i].item()], 
                     "score": round(top_scores[idx].item(), 4)} for idx, i in enumerate(top_idx)]

# --- Usage Example ---
if __name__ == "__main__":
    # Initialize the interface (requires .pth and .pkl files in the same directory)
    interface = GNNInferenceInterface()
    
    # Load input data
    nodes = pd.read_excel('dummy_working.xlsx')
    edges = pd.read_csv('dummy_edges.csv')
    
    # Get Top 10 Prescriptions
    results = interface.prescribe(nodes, edges, top_n=10)
    
    print("\n--- Prescribed Acquisitions ---")
    for r in results:
        print(f"Acquirer: {r['acquirer']} -> Target: {r['target']} (Prob: {r['score']})")