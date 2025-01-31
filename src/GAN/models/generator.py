import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

class ConditionalGraphGenerator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, num_nodes):
        """
        Args:
            noise_dim (int): Dimensionality of the noise vector.
            embedding_dim (int): Dimensionality of the description embeddings.
            num_nodes (int): Number of nodes in the graph.
        """
        super(ConditionalGraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes

        # self.fc0 = nn.Linear(embedding_dim, 10)
        # Fully connected layers for noise and stats fusion
        self.fc1 = nn.Linear(noise_dim + embedding_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_nodes * num_nodes)  # Output: Adjacency matrix

    def forward(self, noise, desc_embd):
        """
        Args:
            noise (torch.Tensor): Random noise of shape (batch_size, noise_dim).
            desc_embd (torch.Tensor): Description embeddings of shape (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Generated adjacency matrices of shape (batch_size, num_nodes, num_nodes).
        """
        # Fully connected layers for description embeddings
        # desc_embd = self.fc0(desc_embd)
        
        # Concatenate noise and stats
        input_data = torch.cat([noise, desc_embd], dim=-1)

        # Fully connected layers
        out = F.relu(self.fc1(input_data))
        out = F.relu(self.fc2(out))
        adj = self.fc3(out)

        # Reshape to adjacency matrix
        adj = adj.view(-1, self.num_nodes, self.num_nodes)

        # Symmetrize
        adj = (adj + torch.transpose(adj, 1, 2)) / 2
        adj = torch.sigmoid(adj)  # Scale values between 0 and 1

        return adj
