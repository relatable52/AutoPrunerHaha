from transformers import AutoModel, T5EncoderModel, PLBartModel
from torch import nn
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.out = nn.Linear(768, 2)

    def forward(self,ids,mask):
        _,emb = self.bert_model(ids,attention_mask=mask, return_dict=False)
        out = self.out(emb)

        return out, emb


class CodeT5Enc(nn.Module):
    def __init__(self):
        super(CodeT5Enc, self).__init__()
        self.codet5_model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask):
        emb = self.codet5_model(input_ids=ids, attention_mask=mask, return_dict=False)[
            0
        ]
        emb = emb[:, -1]
        out = self.out(emb)
        return out, emb


class CodeT5pEnc(nn.Module):
    def __init__(self):
        super(CodeT5pEnc, self).__init__()
        self.codet5p_model = T5EncoderModel.from_pretrained("Salesforce/codet5p-770m")
        self.out = nn.Linear(1024, 2)

    def forward(self, ids, mask):
        emb = self.codet5p_model(input_ids=ids, attention_mask=mask, return_dict=False)[
            0
        ]
        emb = emb[:, -1]
        out = self.out(emb)
        return out, emb


class CodeT5pEmb(nn.Module):
    def __init__(self):
        super(CodeT5pEmb, self).__init__()
        self.codet5p_emb_model = AutoModel.from_pretrained(
            "Salesforce/codet5p-110m-embedding", trust_remote_code=True
        )
        self.out = nn.Linear(256, 2)

    def forward(self, ids, mask):
        emb = self.codet5p_emb_model(input_ids=ids, attention_mask=mask)
        out = self.out(emb)
        return out, emb


class CodeSageBase(nn.Module):
    def __init__(self):
        super(CodeSageBase, self).__init__()
        self.codesage_model = AutoModel.from_pretrained("codesage/codesage-small")
        self.out = nn.Linear(1024, 2)

    def forward(self, ids, mask):
        emb = self.codesage_model(
            input_ids=ids, attention_mask=mask, return_dict=False
        )[1]
        out = self.out(emb)
        return out, emb

class PLBart(nn.Module):
    def __init__(self):
        super(PLBart, self).__init__()
        self.plbart_model = PLBartModel.from_pretrained("uclanlp/plbart-base")
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask):
        emb = self.plbart_model(input_ids = ids, attention_mask=mask)[0][0][0]
        out = self.out(emb)
        return out, emb


def get_model(model_name):
    models_dict = {
        "codebert": BERT,
        "codet5-base": CodeT5Enc,
        "codet5p-770m": CodeT5pEnc,
        "codet5p-110m-embedding": CodeT5pEmb,
        "codesage": CodeSageBase,
        "plbart": PLBart
    }
    if model_name in models_dict:
        model = models_dict[model_name]()
    else:
        return NotImplemented
    return model

def get_emb_size(model_name):
    emb_size_dict = {
        "codebert": 768,
        "codet5-base": 768,
        "codet5p-770m": 1024,
        "codet5p-110m-embedding": 256,
        "codesage": 1024,
        "plbart": 768
    }
    if model_name in emb_size_dict:
        emb_size = emb_size_dict[model_name]
        return emb_size
    raise NotImplemented