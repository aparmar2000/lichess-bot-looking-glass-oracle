# from collections import namedtuple

import torch
import torch.nn as nn
from safetensors import safe_open
from torch.nn import CrossEntropyLoss
from transformers import MambaConfig, MambaModel
from torch._prims_common import DeviceLikeType


class MambaChessMultiHeadsModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.head_1 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=False, **factory_kwargs),
        )
        self.head_2 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=False, **factory_kwargs),
        )
        self.head_3 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=False, **factory_kwargs),
        )
        self.head_4 = nn.Sequential(
            nn.Linear(hidden_size, vocab_size, bias=False, **factory_kwargs),
        )

    def forward(self, hidden_states):
        head_1_logits = self.head_1(hidden_states)
        head_2_logits = self.head_2(hidden_states)
        head_3_logits = self.head_3(hidden_states)
        head_4_logits = self.head_4(hidden_states)
        return (head_1_logits, head_2_logits, head_3_logits, head_4_logits)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MambaChessModel(nn.Module):

    def __init__(
        self,
        config: MambaConfig,
        device:DeviceLikeType|None=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.device = device
        factory_kwargs = {"device": device, "dtype": dtype}

        self.avg_elo = torch.tensor(1500.0)
        self.id_remapper = torch.tensor([*range(78)]+[78,79,80, 78,79,80], dtype=torch.int)

        super().__init__()
        self.backbone: MambaModel = MambaModel(config) \
            .to(device) # type: ignore
        self.chess_head = MambaChessMultiHeadsModule(config.hidden_size, config.vocab_size, **factory_kwargs)

    def forward(self, input_ids, player_1_elo=None, player_2_elo=None, position_ids=None, inference_params=None, num_last_tokens=0, labels=None, label_weights=None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        
        # Normalize player elo
        if player_1_elo is None:
            player_1_elo = torch.tensor([1500]).float().expand(len(input_ids)).unsqueeze(-1).clone()
        if player_2_elo is None:
            player_2_elo = torch.tensor([1500]).float().expand(len(input_ids)).unsqueeze(-1).clone()
        player_1_elo -= self.avg_elo
        player_1_elo /= self.avg_elo
        player_2_elo -= self.avg_elo
        player_2_elo /= self.avg_elo
        
        # Scale player elo tokens
        player_1_elo_mul = ( (input_ids==78) | (input_ids==79) | (input_ids==80) ).float()
        player_1_elo_mul *= player_1_elo
        player_1_elo_mul += player_1_elo_mul==0
        player_2_elo_mul = ( (input_ids==81) | (input_ids==82) | (input_ids==83) ).float()
        player_2_elo_mul *= player_2_elo
        player_2_elo_mul += player_2_elo_mul==0
        
        self.id_remapper = self.id_remapper.to(input_ids.device)
        remapped_input_ids = self.id_remapper[input_ids]
        token_embeddings = self.backbone.get_input_embeddings()(remapped_input_ids)
        token_embeddings *= player_1_elo_mul.unsqueeze(-1).clone()
        token_embeddings *= player_2_elo_mul.unsqueeze(-1).clone()
        
        backbone_output = self.backbone(inputs_embeds=token_embeddings, inference_params=inference_params) #pylint: disable=not-callable
        hidden_states = backbone_output['last_hidden_state']
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        move_logits = self.chess_head(hidden_states)
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(move_logits[0].device).long()
            if label_weights==None:
                label_weights = torch.ones_like(labels)
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct( torch.flatten(move_logits[0], end_dim=1), torch.flatten(labels[:,0:-3], end_dim=1) ) * torch.flatten(label_weights[:,0:-3], end_dim=1)
            loss += loss_fct( torch.flatten(move_logits[1], end_dim=1), torch.flatten(labels[:,1:-2], end_dim=1) ) * torch.flatten(label_weights[:,1:-2], end_dim=1)
            loss += loss_fct( torch.flatten(move_logits[2], end_dim=1), torch.flatten(labels[:,2:-1], end_dim=1) ) * torch.flatten(label_weights[:,2:-1], end_dim=1)
            loss += loss_fct( torch.flatten(move_logits[3], end_dim=1), torch.flatten(labels[:,3:], end_dim=1) ) * torch.flatten(label_weights[:,3:], end_dim=1)
            loss /= 4
            loss = loss.mean()
        
        # ChessOutput = namedtuple("ChessOutput", ["logits", "loss"])
        result = {
            'logits': [*move_logits],
            'loss': loss,
        }
        return result
    
    def num_parameters(self) -> int:
        return self.backbone.num_parameters() + self.chess_head.num_parameters()
    
    def move_context_size(self) -> int:
        return 150
    def token_context_size(self) -> int:
        return (self.move_context_size()*4) + 6
    

def load_from_safetensors(config: MambaConfig, filename: str, device:DeviceLikeType="cpu") -> MambaChessModel:
    # Initializing a model from the configuration
    model = MambaChessModel(config, device=device)

    loaded_dict = {}
    # Load model weights from checkpoint
    with safe_open(filename, framework="pt", device=str(device)) as f: # type: ignore
        for key in f.keys():
            loaded_dict[key] = f.get_tensor(key)
    model.load_state_dict(loaded_dict)
    
    return model