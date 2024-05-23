from torch import nn
from transformers import AutoModel, T5EncoderModel, PLBartModel

models = {
    "codebert-base": {
        "model": AutoModel,
        "max_length": 512,
        "embedding_size": 768,
        "pretrained_name": "microsoft/codebert-base",
    },
    "codet5-base": {
        "model": T5EncoderModel,
        "max_length": 512,
        "embedding_size": 768,
        "pretrained_name": "microsoft/codet5-base",
    },
    "codet5-large": {
        "model": T5EncoderModel,
        "max_length": 512,
        "embedding_size": 768,
        "pretrained_name": "microsoft/codet5-large",
    },
    "plbart-base": {
        "model": PLBartModel,
        "max_length": 1024,
        "embedding_size": 768,
        "pretrained_name": "uclanlp/plbart-base",
    },
    "plbart-large": {
        "model": PLBartModel,
        "max_length": 1024,
        "embedding_size": 1024,
        "pretrained_name": "uclanlp/plbart-large",
    },
    "codet5p-base": {
        "model": T5EncoderModel,
        "max_length": 512,
        "embedding_size": 256,
        "pretrained_name": "Salesforce/codet5p-110m-embedding",
        "trust_remote_code": True,
    },
    "codet5p-large": {
        "model": T5EncoderModel,
        "max_length": 512,
        "embedding_size": 768,
        "pretrained_name": "Salesforce/codet5p-770m",
        "trust_remote_code": True,
    },
    "codesage-base": {
        "model": AutoModel,
        "max_length": 2048,
        "embedding_size": 1024,
        "pretrained_name": "codesage/codesage-base",
        "trust_remote_code": True,
    },
    "codesage-small": {
        "model": AutoModel,
        "max_length": 2048,
        "embedding_size": 1024,
        "pretrained_name": "codesage/codesage-small",
        "trust_remote_code": True,
    }
}


class EmbeddingModel(nn.Module):
    def __init__(self, model_name):
        assert model_name in models, f"Model name {model_name} not found"
        super(EmbeddingModel, self).__init__()
        self.model_name = model_name
        self.encoder = self.load_model()
        emb_size = models[self.model_name]["embedding_size"]
        self.fc = nn.Linear(emb_size, 2)

    def forward(self, ids, mask):
        if self.model_name == "codet5p-base":
            emb = self.encoder(ids, attention_mask=mask)
        else:
            emb = self.encoder(ids, attention_mask=mask, return_dict=False)[0][:, 0]
        out = self.fc(emb)
        return out, emb

    def load_model(self):
        if "trust_remote_code" in models[self.model_name]:
            return models[self.model_name]["model"].from_pretrained(
                models[self.model_name]["pretrained_name"],
                trust_remote_code=True,
            )
        return models[self.model_name]["model"].from_pretrained(
            models[self.model_name]["pretrained_name"]
        )
