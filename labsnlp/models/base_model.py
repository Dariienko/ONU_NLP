import torch

class TextClassificationModdule(torch.nn.Module):
    def __init__(
        self,
        encoder:  torch.nn.Module,
        classifier: torch.nn.Module
    ):
        super().__init__()
        self.encoder = encoder  # hidden_states, (h_t, c_t)
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.encoder(x)[1][0])