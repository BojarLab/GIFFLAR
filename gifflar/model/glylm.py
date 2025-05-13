from pathlib import Path
import torch
from transformers import EsmModel

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN
from gifflar.model.utils import get_prediction_head
from gifflar.tokenize.tokenizer import GIFFLARTokenizer
from gifflar.train_lm import PRE_TOKENIZERS

def pipeline(tokenizer, glycan_lm, iupac):
    tokens = tokenizer(iupac)
    input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(glycan_lm.device)[:, :606]
    attention = torch.tensor(tokens["attention_mask"]).unsqueeze(0).to(glycan_lm.device)[:, :606]
    with torch.no_grad():
        return glycan_lm(input_ids, attention).last_hidden_state


class GlycanLM(DownstreamGGIN):
    def __init__(self, token_file, model_dir, hidden_dim: int, pre_tokenizer: str, *args, **kwargs):
        super().__init__(feat_dim=1, hidden_dim=hidden_dim, *args, **kwargs)
        del self.convs

        pretokenizer = PRE_TOKENIZERS[pre_tokenizer]()
        
        if Path(token_file).exists():
            mode = "BPE" if "bpe" in token_file else "WP"
            tokenizer = GIFFLARTokenizer(pretokenizer, mode)
            tokenizer.load(token_file)
        else:
            tokenizer = GIFFLARTokenizer(pretokenizer, "NONE")
            tokenizer.load(None)
        
        glycan_lm = EsmModel.from_pretrained(model_dir)
        self.encoder = lambda x: pipeline(tokenizer, glycan_lm, x)
        self.head, self.loss, self.metrics = get_prediction_head(hidden_dim, self.output_dim, self.task)

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Make predictions based on the molecular fingerprint.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings (None for the MLP), the graph embedding, and the final model prediction
        """
        graph_embeddings = torch.cat([self.encoder(iupac).mean(dim=1) for iupac in batch["IUPAC"]], dim=0).to(self.device)
        return {
            "node_embed": None,
            "graph_embed": graph_embeddings,
            "preds": self.head(graph_embeddings),
        }
