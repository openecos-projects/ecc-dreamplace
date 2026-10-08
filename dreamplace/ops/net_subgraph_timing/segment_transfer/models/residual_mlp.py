import torch


class ResidualSegmentTransferMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, features):
        return self.net(features)
