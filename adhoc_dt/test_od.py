import torch
import unittest
from Networks.ODITSEncoder import TeamworkSituationEncoder, ProxyEncoder
from Networks.ODITSDecoder import TeamworkSituationDecoder, IntegratingNet, ProxyDecoder, MarginalUtilityNet

class TestODITSModules(unittest.TestCase):
    def setUp(self):
        # Test parameters
        self.batch_size = 256
        self.state_dim = 75
        self.action_dim = 5
        self.hidden_dim = 128
        self.output_dim = 32
        self.num_agents = 2
        
        # Initialize encoder and decoder
        self.teamwork_encoder = TeamworkSituationEncoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.proxy_encoder = ProxyEncoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Initialize decoders
        primary_layer_dims = [(self.hidden_dim, 1)]
        self.teamwork_decoder = TeamworkSituationDecoder(
            hyper_input_dim=self.output_dim,
            primary_layer_dims=primary_layer_dims
        )
        
        self.proxy_decoder = ProxyDecoder(
            hyper_input_dim=self.output_dim,
            primary_layer_dims=primary_layer_dims
        )
        
        self.integrating_net = IntegratingNet(
            input_dim=1,  # Changed to 1 since input is marginal utility
            hidden_dim=self.hidden_dim,
            fc_input_dim=self.hidden_dim,
            fc_output_dim=1,
            hypernetwork=self.teamwork_decoder
        )
        
        self.marginal_net = MarginalUtilityNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            fc_input_dim=self.hidden_dim,
            fc_output_dim=1,
            hypernetwork=self.proxy_decoder
        )

    def test_teamwork_encoder_decoder(self):
        # Create test inputs
        states = torch.randn(self.batch_size, self.state_dim * self.num_agents)
        actions = torch.randn(self.batch_size, self.action_dim * self.num_agents)
        
        # Test encoder
        mean, log_var = self.teamwork_encoder(states, actions)
        
        # Test shapes
        self.assertEqual(mean.shape, (self.batch_size, self.output_dim))
        self.assertEqual(log_var.shape, (self.batch_size, self.output_dim))
        
        # Sample from distribution
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        
        # First get marginal utility
        x = torch.randn(self.batch_size, 1, self.state_dim + self.action_dim)
        h_0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        marginal_output = self.marginal_net(x, z, h_0)  # (batch_size, 1)
        
        # Test integrating net with marginal utility as input
        output = self.integrating_net(marginal_output, z)
        
        self.assertEqual(output.shape, (self.batch_size, 1, 1))

    def test_proxy_encoder_decoder(self):
        # Create test inputs
        states = torch.randn(self.batch_size, self.state_dim)
        actions = torch.randn(self.batch_size, self.action_dim)
        
        # Test encoder
        mean, log_var = self.proxy_encoder(states, actions)
        
        # Test shapes
        self.assertEqual(mean.shape, (self.batch_size, self.output_dim))
        self.assertEqual(log_var.shape, (self.batch_size, self.output_dim))
        
        # Sample from distribution
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        
        # Test marginal utility net
        x = torch.randn(self.batch_size, 1, self.state_dim + self.action_dim)
        h_0 = torch.zeros(1, self.batch_size, self.hidden_dim)
        output = self.marginal_net(x, z, h_0)
        
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_value_ranges(self):
        # Test inputs
        states = torch.randn(self.batch_size, self.state_dim)
        actions = torch.randn(self.batch_size, self.action_dim)
        
        # Test proxy encoder outputs
        mean, log_var = self.proxy_encoder(states, actions)
        
        # Check if values are finite
        self.assertTrue(torch.all(torch.isfinite(mean)))
        self.assertTrue(torch.all(torch.isfinite(log_var)))
        
        # Check if log_var is not too extreme
        self.assertTrue(torch.all(log_var > -20))
        self.assertTrue(torch.all(log_var < 20))

if __name__ == '__main__':
    unittest.main()
