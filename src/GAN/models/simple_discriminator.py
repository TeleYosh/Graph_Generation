import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGraphDiscriminator(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim=128):
        """
        Args:
            num_nodes (int): Number of nodes in the graph.
            conditional_dim (int): Dimensionality of the conditioning vector (stats).
            hidden_dim (int): Hidden dimensionality for MLP layers.
        """
        super(ConditionalGraphDiscriminator, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        # self.fc0 = nn.Linear(embedding_dim, 10)
        
        # Fully connected layers for adjacency and embedding fusion
        self.fc1 = nn.Linear(num_nodes * num_nodes + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.a = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, adj, desc_embd):
        """
        Args:
            adj (torch.Tensor): Adjacency matrices of shape (batch_size, num_nodes, num_nodes).
            desc_embd (torch.Tensor): Description embeddings of shape (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Real/fake scores of shape (batch_size, 1).
        """
        
        # Fully connected layers for description embeddings
        # desc_embd = self.fc0(desc_embd)
                             
        # Flatten adjacency matrix
        adj_flat = adj.view(adj.size(0), -1)  # Shape: (batch_size, num_nodes * num_nodes)

        # Concatenate adjacency with stats
        input_data = torch.cat([adj_flat, desc_embd], dim=-1)


        # Fully connected layers
        out = self.fc1(input_data)
        out = self.a(out)
        out = self.fc2(out)
        score = torch.sigmoid(out)  # Scale between 0 and 1

        return score
