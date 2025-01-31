import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGraphDiscriminator(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim=128, num_layers=3):
        """
        Args:
            num_nodes (int): Number of nodes in the graph.
            embedding_dim (int): Dimensionality of the conditioning vector (stats).
            hidden_dim (int): Hidden dimensionality for MLP layers.
            num_layers (int): Number of GCN layers.
        """
        super(ConditionalGraphDiscriminator, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # List of Graph Convolutional Layers with residual connections
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(
                nn.Linear(num_nodes, num_nodes)
            )

        # Final fully connected layer for scalar output
        self.fc0 = nn.Linear(num_nodes*num_nodes, 2*hidden_dim)
        self.fc1 = nn.Linear(2*hidden_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()

    def forward(self, adj, desc_embd):
        """
        Args:
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).
            desc_embd (torch.Tensor): Description embeddings of shape (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Real/fake scores of shape (batch_size, 1).
        """
        
        batch_size = adj.size(0)

        adj = adj + torch.eye(self.num_nodes, device=adj.device).unsqueeze(0)
        

        degree = torch.sum(adj, dim=1)
        degree = degree + 1e-8
        D_inv_sqrt = torch.sqrt(1.0 / degree)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
        
        if torch.isnan(adj_normalized).any() or torch.isinf(adj_normalized).any():
            raise ValueError("NaN or Inf in normalized adjacency matrix")
        
        # Initial graph embedding is the identity matrix
        H = torch.eye(self.num_nodes, device=adj.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # GCN layers with residual connections
        for layer in self.gcn_layers:
            H_new = self.activation(layer(torch.matmul(adj_normalized, H)))
            H = H + H_new

        # Flatten the final graph embedding and concatenate with conditional embeddings
        H_final = H.view(batch_size, -1) 
        
        h = self.fc0(H_final)
        h = torch.cat([h, desc_embd], dim=-1)  

        # Final scalar output
        out = self.fc1(h)
        out = self.activation(out)
        score = torch.sigmoid(out)

        return score

