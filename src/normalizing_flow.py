import torch
import torch.nn as nn


"""
Implementation of conditional normalizing flows. 
The conditioning mechanism is applied by modifying the scale and translation networks 
of the affine coupling layers to incorporate embedded graph properties.

Affine coupling layer: https://arxiv.org/pdf/1605.08803 (without conditioning)
Conditional coupling: https://arxiv.org/pdf/1912.00042
"""


class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, d_cond):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.dim = dim

        # MLPs for scale and shift computation
        self.scale_mlp = nn.Sequential(
            nn.Linear(dim // 2 + d_cond, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh(),  # To keep scale between -1 and 1
        )
        self.shift_mlp = nn.Sequential(
            nn.Linear(dim // 2 + d_cond, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
        )

    def forward(self, x, c):
        """
        Forward pass of the conditional affine coupling layer:
        - Split the input x into two parts x_a and x_b
        - Compute scale and shift using x_a and the conditioning c
        - Apply the affine transformation to x_b
        - Return the transformed x and the log determinant of the Jacobian
        """
        x_a, x_b = x[:, : self.dim // 2], x[:, self.dim // 2 :]
        x_a_c = torch.cat([x_a, c], dim=1)
        scale = self.scale_mlp(x_a_c)
        shift = self.shift_mlp(x_a_c)
        z_b = x_b * torch.exp(scale) + shift
        log_det_jacobian = torch.sum(scale, dim=1)
        return torch.cat([x_a, z_b], dim=1), log_det_jacobian

    def inverse(self, z, c):
        """
        Inverse pass of the conditional affine coupling layer.
        """
        z_a, z_b = z[:, : self.dim // 2], z[:, self.dim // 2 :]
        z_a_c = torch.cat([z_a, c], dim=1)
        scale = self.scale_mlp(z_a_c)
        shift = self.shift_mlp(z_a_c)
        x_b = (z_b - shift) * torch.exp(-scale)
        x = torch.cat([z_a, x_b], dim=1)
        return x


class ConditionalNormalizingFlow(nn.Module):
    def __init__(
        self, dim, hidden_dim, n_cond, d_cond, flow_length, alternating_pattern="odd_even"
    ):
        super(ConditionalNormalizingFlow, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_cond = n_cond
        self.flows = nn.ModuleList(
            [
                ConditionalAffineCouplingLayer(dim, hidden_dim, d_cond)
                for _ in range(flow_length)
            ]
        )
        # MLP for conditioning (to embed graph properties)
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )
        # Alternating patterns (to ensure all dimensions are transformed)
        if alternating_pattern == "odd_even":
            self.permutation = self._odd_even_pattern
            assert (
                dim % 2 == 0
            ), "odd_even method requires an even input dimension"
        elif alternating_pattern == "flip":
            self.permutation = self._flip_pattern
        else:
            raise ValueError("Invalid alternating_pattern input. Use 'flip' or 'even_odd'")
        
        # self._assert_invertibility()

    def _odd_even_pattern(self, x, reverse=False):
        if reverse:
            n = x.size(1) // 2
            x_a, x_b = x[:, :n], x[:, n:]
            x_combined = torch.empty_like(x)
            x_combined[:, ::2] = x_a
            x_combined[:, 1::2] = x_b
            return x_combined
        return torch.cat([x[:, ::2], x[:, 1::2]], dim=1)

    def _flip_pattern(self, x, reverse=False):
        return x.flip(1)

    def forward(self, x, c):
        """
        Forward pass of the conditional normalizing flow.
        Apply dimension permutation before each flow.
        """
        c = self.cond_mlp(c)

        log_det_jacobian = 0
        z = x
        for flow in self.flows:
            z = self.permutation(z)
            z, log_dj = flow(z, c)
            log_det_jacobian += log_dj
        return z, log_det_jacobian

    def inverse(self, z, c):
        c = self.cond_mlp(c)

        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x, c)
            x = self.permutation(x, reverse=True)

        return x

    def p_losses(self, x, c):
        """
        Loss (negative log-likelihood) on a batch of data
        """
        # In the case of a Gaussian base distribution, the log-likelihood is the negative squared error (without considering the constant term)
        z, log_det_jacobian = self.forward(x, c)
        loss = torch.sum(0.5 * z**2, dim=1) - log_det_jacobian
        return torch.mean(loss)

    @torch.no_grad()
    def sample(self, c, n_samples):
        """
        Sample from the model
        """
        z = torch.randn(n_samples, self.dim)
        return self.inverse(z, c)

    @torch.no_grad()
    def _assert_invertibility(self, n_samples=2):
        x = torch.randn(n_samples, self.dim)
        c = torch.randn(n_samples, self.n_cond)
        z, _ = self.forward(x, c)
        x_reconstructed = self.inverse(z, c)
        assert torch.allclose(x, x_reconstructed, atol=1e-6), "Model is not invertible"
        print("Model is invertible")
