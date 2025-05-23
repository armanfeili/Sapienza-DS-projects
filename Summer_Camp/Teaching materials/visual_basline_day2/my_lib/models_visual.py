import timm
import torch.nn as nn
import torch


class VisualModelTimm(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, num_classes=0, pretrained=pretrained
        )
        config = timm.get_pretrained_cfg(model_name=model_name, allow_unregistered=True)
        config = config.to_dict()
        with torch.no_grad():
            emb = self.encoder(torch.rand(1, *config["input_size"]))
            self.embedding_size = emb.shape[1]

        self.head = nn.Linear(self.embedding_size, num_classes)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encode(x)
        return self.head(x)
