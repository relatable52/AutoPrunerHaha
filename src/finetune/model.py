from torch import nn
from transformers import AutoModel, T5EncoderModel, PLBartModel

models = {
    "codebert": {
        "model": AutoModel,
        "embedding_size": 768,
        "max_length": 512,
        "pretrained_name": {
            "base": "microsoft/codebert-base",
        },
    },
    "codet5": {
        "model": T5EncoderModel,
        "embedding_size": 768,
        "max_length": 512,
        "pretrained_name": {
            "base": "Salesforce/codet5-base",
            "large": "Salesforce/codet5-large",
        },
    },
    "plbart": {
        "model": PLBartModel,
        "embedding_size": 768,
        "max_length": 1024,
        "pretrained_name": {
            "base": "uclanlp/plbart-base",
        },
    },
    "codet5p": {
        "model": T5EncoderModel,
        "embedding_size": 768,
        "max_length": 512,
        "pretrained_name": {
            "base": "Salesforce/codet5p-110m-embedding",
            "large": "Salesforce/codet5p-770m",
        },
    },
    "codesage": {
        "model": AutoModel,
        "embedding_size": 1024,
        "max_length": 2048,
        "pretrained_name": {
            "small": "microsoft/codesage-small",
            "base": "microsoft/codesage-base",
        },
    },
}


class Embedding(nn.Module):
    def __init__(self, model_name, model_size):
        assert self.model_name not in models, f"Model name {self.model_name} not found"
        assert (
            self.embed_size not in models[self.model_name]["size"]
        ), f"Model size {self.model_size} not found in {self.model_name}"
        super(Embedding, self).__init__()
        self.name = model_name
        self.model_size = model_size
        self.encoder = self.load_model()
        emb_size = models[self.model_name]["size"][self.model_size]["embedding_size"]
        self.fc = nn.Linear(emb_size, 2)

    def forward(self, ids, mask):
        emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0][:, 0, :]
        out = self.fc(emb)
        return out, emb

    def load_model(self):
        return models[self.model_name]["model"].from_pretrained(
            models[self.model_name]["size"][self.model_size]["pretrained_name"]
        )
