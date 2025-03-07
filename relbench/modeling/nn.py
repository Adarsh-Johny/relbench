from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.data import HeteroData
from torch_frame import stype
from torch_frame.nn import (
    EmbeddingEncoder,
    LinearEmbeddingEncoder,
    LinearEncoder,
    MultiCategoricalEmbeddingEncoder,
    TimestampEncoder,
    TabTransformer,
)

class HeteroEncoder(nn.Module):
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[str, list[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_to_stype_dict: Dict[NodeType, Dict[str, 'stype']],
        torch_frame_model_cls=TabTransformer,
        torch_frame_model_kwargs: Dict[str, Any] | None = None,
        default_stype_encoder_cls_kwargs: Dict[str, Tuple[type, Dict[str, Any]]] | None = None,
    ):
        super().__init__()
        if default_stype_encoder_cls_kwargs is None:
            default_stype_encoder_cls_kwargs = {
                stype.categorical: (EmbeddingEncoder, {}),
                stype.numerical: (LinearEncoder, {}),
                stype.multicategorical: (MultiCategoricalEmbeddingEncoder, {}),
                stype.embedding: (LinearEmbeddingEncoder, {}),
                stype.timestamp: (TimestampEncoder, {}),
                stype.text_embedded: (LinearEmbeddingEncoder, {}),
            }
        if torch_frame_model_kwargs is None:
            torch_frame_model_kwargs = {}

        # Default values for TabTransformer if not provided in kwargs
        default_kwargs = {
            "out_channels": channels,  # Match GNN input size
            "num_layers": 2,          # Reasonable default for transformer layers
            "num_heads": 4,           # Reasonable default for attention heads
            "encoder_pad_size": 0,    # No padding by default (adjust if needed)
            "attn_dropout": 0.1,      # Standard dropout rate
            "ffn_dropout": 0.1,       # Standard dropout rate
        }
        # Update defaults with any user-provided kwargs
        torch_frame_model_kwargs = {**default_kwargs, **torch_frame_model_kwargs}

        self.encoders = torch.nn.ModuleDict()
        for node_type in node_to_col_names_dict.keys():
            col_names_dict = node_to_col_names_dict[node_type]
            torch_frame_model = torch_frame_model_cls(
                channels=channels,              # Input embedding size
                col_stats=node_to_col_stats[node_type],
                col_names_dict=col_names_dict,
                out_channels=torch_frame_model_kwargs["out_channels"],
                num_layers=torch_frame_model_kwargs["num_layers"],
                num_heads=torch_frame_model_kwargs["num_heads"],
                encoder_pad_size=torch_frame_model_kwargs["encoder_pad_size"],
                attn_dropout=torch_frame_model_kwargs["attn_dropout"],
                ffn_dropout=torch_frame_model_kwargs["ffn_dropout"],
                # Pass any additional kwargs that might be supported
                **{k: v for k, v in torch_frame_model_kwargs.items() if k not in default_kwargs},
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(self, tf_dict: Dict[NodeType, "TensorFrame"]) -> Dict[NodeType, torch.Tensor]:
        out_dict = {}
        for node_type, tf in tf_dict.items():
            out_dict[node_type] = self.encoders[node_type](tf)
        return out_dict

class LSTMBasedTemporalEncoder(nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()
        self.lstm_dict = nn.ModuleDict({node: nn.LSTM(input_size=channels, hidden_size=channels, batch_first=True) for node in node_types})

    def forward(self, h_dict: Dict[NodeType, Tensor], time_dict: Dict[NodeType, Tensor], batch_dict: Dict[NodeType, Tensor]) -> Dict[NodeType, Tensor]:
        updated_h_dict = {}
        for node_type, lstm in self.lstm_dict.items():
            if node_type in h_dict and h_dict[node_type].size(0) > 0:
                lstm_input = h_dict[node_type].unsqueeze(0)  # Add batch dimension
                _, (hn, _) = lstm(lstm_input)
                updated_h_dict[node_type] = hn.squeeze(0)  # Remove batch dimension
        return updated_h_dict
    
    def reset_parameters(self):
        """Reset the parameters of all LSTM modules in lstm_dict."""
        for lstm in self.lstm_dict.values():
            lstm.reset_parameters()

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {edge_type: SAGEConv((channels, channels), channels, aggr=aggr) for edge_type in edge_types},
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Tensor]) -> Dict[NodeType, Tensor]:
        for conv, norm_dict in zip(self.convs, self.norms):
            # Apply the heterogeneous convolution.
            out_dict = conv(x_dict, edge_index_dict)
            # Ensure that every node type present in x_dict has a valid output.
            for node_type in x_dict.keys():
                if node_type not in out_dict or out_dict[node_type] is None:
                    # Replace with the original features or zeros with the correct shape.
                    num_nodes = x_dict[node_type].size(0)
                    out_dict[node_type] = x_dict[node_type]
                    # Alternatively, if you prefer zeros:
                    # out_dict[node_type] = torch.zeros((num_nodes, self.gnn.channels), device=x_dict[node_type].device)
            # Apply normalization and activation.
            x_dict = {key: norm_dict[key](out_dict[key]) for key in out_dict.keys()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict


class SnapshotTemporalGNN(nn.Module):
    def __init__(self, node_types, edge_types, input_dim, hidden_dim, num_layers):
        super(SnapshotTemporalGNN, self).__init__()
        self.gnn = HeteroGraphSAGE(node_types, edge_types, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, snapshots: List[HeteroData]):
        x_list = []
        for snapshot in snapshots:
            x_dict = self.gnn(snapshot.x_dict, snapshot.edge_index_dict)
            x_list.append(torch.cat([x for x in x_dict.values()], dim=0))
        x_seq = torch.stack(x_list, dim=1)
        output, _ = self.lstm(x_seq)
        return output[:, -1, :]