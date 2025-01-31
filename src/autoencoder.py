import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool


"""
Code from the baseline implementation (`autoencoder.py`).
The encoder was modified to use a GAT layer instead of a GIN layer.
The decoder was not modified.
The architecture was modified to include a conditional embedding.
"""



# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


# Encoder 
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=1, concat=False))
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out




# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, n_condition=7, keep_exact_cond=True, alt_loss=False):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GAT(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)

        # Conditional embedding
        self.label_projector = nn.Sequential(
            nn.Linear(n_condition, latent_dim),
            nn.ReLU(),
        )
        # Conditionning (exact properties values or embeddings with the same dim as latent_dim)
        cond_dim = n_condition if keep_exact_cond else latent_dim
        self.keep_exact_cond = keep_exact_cond
        
        # Decoder
        self.decoder = Decoder(latent_dim + cond_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

        # Loss parameters
        self.beta = 0
        self.weights = {
            'nodes': 1.0, 
            'edges': 3.0, 
            'triangles': 3.0,   
            'degree': 0.5       
        }
        self.alt_loss = alt_loss
        
    def _get_condition_embedding(self, c, keep_exact_cond=True):
        if not keep_exact_cond:
            c = self.label_projector(c)
        return c
    
    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def forward(self, data):
        x_g = self.encode(data)
        c = self._get_condition_embedding(data.stats, self.keep_exact_cond)
        x_f = torch.cat((x_g, c), dim=1) # concat conditionning
        adj = self.decoder(x_f)
        return adj

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x, c):
       c = self._get_condition_embedding(c, self.keep_exact_cond)
       x_f = torch.cat((x,c), dim=1)
       adj = self.decoder(x_f)
       return adj

    def loss_function(self, data, current_epoch, warmup_epochs=10, max_beta=0.1, alpha=0.01):
        # Encode
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        
        # Conditionning
        c = self._get_condition_embedding(data.stats, self.keep_exact_cond)
        x_f = torch.cat((x_g, c), dim=1) # concat cond

        # Decode
        adj = self.decoder(x_f)

        # Losses
        beta = min(current_epoch / warmup_epochs, 1.0)*max_beta
        self.beta = beta
        # recon = F.l1_loss(adj, data.A, reduction='mean')
        recon = F.huber_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        stats_loss = 0 if not self.alt_loss else self._compute_statistics_loss(adj, data.stats, self.weights)
        loss = recon + beta*kld + alpha*stats_loss

        return loss, recon, kld

    def _compute_statistics_loss(self, A_pred, stats_true, weights):
        # Compute graph statistics for predicted adjacency matrices
        num_nodes_pred = (A_pred.sum(dim=2) > 0).sum(dim=1)
        num_edges_pred = A_pred.sum(dim=(1, 2)) / 2
        triangles_pred = torch.diagonal(torch.matrix_power(A_pred, 3), 0, dim1=1, dim2=2).sum(dim=1) / (2*3)
        avg_degree_pred = 2*num_edges_pred / num_nodes_pred
        
        # Ground truth statistics
        num_nodes_true = stats_true[:,0]
        num_edges_true = stats_true[:,1]
        triangles_true = stats_true[:,3]
        avg_degree_true = stats_true[:,2]
    
        # Compute weighted loss
        loss_nodes = F.l1_loss(num_nodes_pred, num_nodes_true)
        loss_edges = F.l1_loss(num_edges_pred, num_edges_true)
        loss_triangles = F.l1_loss(triangles_pred, triangles_true)
        loss_degree = F.l1_loss(avg_degree_pred, avg_degree_true)
    
        loss = (
            weights['nodes'] * loss_nodes +
            weights['edges'] * loss_edges +
            weights['triangles'] * loss_triangles +
            weights['degree'] * loss_degree
        )
        
        return loss
    


@torch.no_grad()
def sample_from_latent_space(vae, latent_dim, c, device):
    # Generate random latent vectors
    bs = c.shape[0]
    z = torch.randn(bs, latent_dim).to(device) # Sampling from N(0, I)
    # Decode the random vectors
    generated_samples = vae.decode(z, c)
    return generated_samples
