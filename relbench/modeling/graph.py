from datetime import datetime
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time
from typing import Any, Dict, List, Optional, Tuple


# def make_pkey_fkey_graph(
#     db: Database,
#     col_to_stype_dict: Dict[str, Dict[str, stype]],
#     text_embedder_cfg: Optional[TextEmbedderConfig] = None,
#     cache_dir: Optional[str] = None,
# ) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
#     r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
#     foreign key relationships, together with the column stats of each table.

#     Args:
#         db: A database object containing a set of tables.
#         col_to_stype_dict: Column to stype for
#             each table.
#         text_embedder_cfg: Text embedder config.
#         cache_dir: A directory for storing materialized tensor
#             frames. If specified, we will either cache the file or use the
#             cached file. If not specified, we will not use cached file and
#             re-process everything from scratch without saving the cache.

#     Returns:
#         HeteroData: The heterogeneous :class:`PyG` object with
#             :class:`TensorFrame` feature.
#     """
#     data = HeteroData()
#     col_stats_dict = dict()
#     if cache_dir is not None:
#         os.makedirs(cache_dir, exist_ok=True)

#     for table_name, table in db.table_dict.items():
#         # Materialize the tables into tensor frames:
#         df = table.df
#         # Ensure that pkey is consecutive.
#         if table.pkey_col is not None:
#             assert (df[table.pkey_col].values == np.arange(len(df))).all()

#         col_to_stype = col_to_stype_dict[table_name]

#         # Remove pkey, fkey columns since they will not be used as input
#         # feature.
#         remove_pkey_fkey(col_to_stype, table)

#         if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
#             col_to_stype = {"__const__": stype.numerical}
#             # We need to add edges later, so we need to also keep the fkeys
#             fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
#             df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

#         path = (
#             None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
#         )

#         dataset = Dataset(
#             df=df,
#             col_to_stype=col_to_stype,
#             col_to_text_embedder_cfg=text_embedder_cfg,
#         ).materialize(path=path)

#         data[table_name].tf = dataset.tensor_frame
#         col_stats_dict[table_name] = dataset.col_stats

#         # Add time attribute:
#         if table.time_col is not None:
#             data[table_name].time = torch.from_numpy(
#                 to_unix_time(table.df[table.time_col])
#             )

#         # Add edges:
#         for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
#             pkey_index = df[fkey_name]
#             # Filter out dangling foreign keys
#             mask = ~pkey_index.isna()
#             fkey_index = torch.arange(len(pkey_index))
#             # Filter dangling foreign keys:
#             pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
#             fkey_index = fkey_index[torch.from_numpy(mask.values)]
#             # Ensure no dangling fkeys
#             assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

#             # fkey -> pkey edges
#             edge_index = torch.stack([fkey_index, pkey_index], dim=0)
#             edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
#             data[edge_type].edge_index = sort_edge_index(edge_index)

#             # pkey -> fkey edges.
#             # "rev_" is added so that PyG loader recognizes the reverse edges
#             edge_index = torch.stack([pkey_index, fkey_index], dim=0)
#             edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
#             data[edge_type].edge_index = sort_edge_index(edge_index)

#     data.validate()

#     return data, col_stats_dict

__all__ = ["make_snapshot_graph"]

def generate_timestamps(main_table, interval_days: int) -> List[pd.Timestamp]:
    """Generates a list of timestamps based on the first and last date of the main table."""
    main_df = main_table.df.copy()
    main_df[main_table.time_col] = pd.to_datetime(main_df[main_table.time_col])
    
    first_date = main_df[main_table.time_col].min()
    last_date = main_df[main_table.time_col].max()

    timestamps = pd.date_range(start=first_date, end=last_date, freq=f"{interval_days}D").to_list()
    
    if not timestamps:
        print("No valid timestamps found!")
    
    return timestamps

def process_main_tables(db, col_to_stype_dict, snapshot, ts, text_embedder_cfg):
    """Processes main tables by filtering data up to the given timestamp."""
    col_stats_dict = {}
    
    for table_name, table in db.table_dict.items():
        if table.time_col is None or table.time_col not in table.df.columns:
            continue  # Skip tables without a time column

        df = table.df[pd.to_datetime(table.df[table.time_col]) <= ts]  # Filter by time
        if df.empty:
            continue  # Skip if no data available
        
        col_to_stype = col_to_stype_dict[table_name]
        
        col_to_stype_copy = col_to_stype.copy()
        remove_pkey_fkey(col_to_stype_copy, table)  # Work on the copy, not the original

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,  # Added text embedder
        ).materialize()

        snapshot[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute
        snapshot[table_name].time = torch.from_numpy(to_unix_time(df[table.time_col]))

    return col_stats_dict

def safe_convert_timestamp(ts_array):
    """Safely converts timestamp arrays to datetime, avoiding invalid dates."""
    valid_dates = []
    for ts in ts_array:
        try:
            valid_dates.append(datetime(*map(int, ts[:6])))
        except ValueError as e:
            print(f"Skipping invalid date {ts[:6]} -> {e}")
            valid_dates.append(None)  # Mark as None for now
    return valid_dates

def process_related_tables(db, col_to_stype_dict, snapshot, text_embedder_cfg):
    """Processes related tables by including only referenced entities in the snapshot.
    
    Ensures tables are added even if they don’t have FK relationships -- but their PK is referenced elsewhere.
    """
    col_stats_dict = {}

    print("\n=== Processing Related Tables ===")  

    referenced_tables = set()  

    for table_name, table in db.table_dict.items():
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            if pkey_table_name in snapshot.to_dict():
                referenced_tables.add(pkey_table_name)  

    for table_name, table in db.table_dict.items():
        if table_name in snapshot.to_dict():
            continue

        df = table.df.copy()
        initial_row_count = len(df)

        print(f"\n Checking table: {table_name}")
        print(f"   -> Initial row count: {initial_row_count}")
        print(f"  -> Foreign keys: {table.fkey_col_to_pkey_table}")

        referenced = False  

        # Step 1: Check FK → PK relationships
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            if pkey_table_name in snapshot.to_dict() and fkey_name in df.columns:
                tf = snapshot[pkey_table_name].tf
                feat_dict = tf.feat_dict  

                # Extract numerical features
                num_data = {
                    col: feat_dict[stype.numerical][:, i].cpu().numpy() 
                    for i, col in enumerate(tf.col_names_dict.get(stype.numerical, []))
                }

                # Extract categorical features
                cat_data = {
                    col: feat_dict[stype.categorical][:, i].cpu().numpy() 
                    for i, col in enumerate(tf.col_names_dict.get(stype.categorical, []))
                }

                # Extract timestamp features
                if stype.timestamp in feat_dict:
                    timestamp_tensor = feat_dict[stype.timestamp].cpu().numpy()
                    time_cols = tf.col_names_dict.get(stype.timestamp, [])
                    
                    time_data = {
                        col: safe_convert_timestamp(timestamp_tensor[:, i, :])
                        for i, col in enumerate(time_cols)
                    }
                else:
                    time_data = {}

                # Extract embeddings
                if stype.embedding in feat_dict:
                    embedding_tensor = feat_dict[stype.embedding].values  
                    embedding_dim = embedding_tensor.shape[1]  

                    embedding_data = {
                        f"embedding_{i}": embedding_tensor[:, i].cpu().numpy()
                        for i in range(embedding_dim)
                    }
                else:
                    embedding_data = {}

                # Convert to Pandas DataFrame
                main_df = pd.DataFrame({**num_data, **cat_data, **time_data, **embedding_data})
                print(f"Converted `{pkey_table_name}` TensorFrame to Pandas DataFrame")

                # Get correct primary key column
                pkey_column = db.table_dict[pkey_table_name].pkey_col
                if pkey_column not in main_df.columns:
                    raise KeyError(f"Primary key column '{pkey_column}' not found in `{pkey_table_name}` DataFrame")

                # Filter by correct primary key
                before_filter = len(df)
                df = df[df[fkey_name].isin(main_df[pkey_column])]
                after_filter = len(df)

                if after_filter > 0:
                    referenced = True
                print(f"   Matched rows: {before_filter} → {after_filter}")

        # Step 2: Check if PK of this table is referenced elsewhere (indirect links)
        if not referenced:
            for other_table_name, other_table in db.table_dict.items():
                if other_table_name in snapshot.to_dict():
                    for other_fkey, other_pkey in other_table.fkey_col_to_pkey_table.items():
                        tf = snapshot[other_table_name].tf
                        feat_dict = tf.feat_dict  

                        # Convert TensorFrame to DataFrame
                        num_data = {
                            col: feat_dict[stype.numerical][:, i].cpu().numpy() 
                            for i, col in enumerate(tf.col_names_dict.get(stype.numerical, []))
                        }
                        cat_data = {
                            col: feat_dict[stype.categorical][:, i].cpu().numpy() 
                            for i, col in enumerate(tf.col_names_dict.get(stype.categorical, []))
                        }

                        ref_df = pd.DataFrame({**num_data, **cat_data})

                        if other_pkey == table_name and other_fkey in ref_df.columns:
                            referenced = True
                            print(f" `{table_name}` is referenced as a PK in `{other_table_name}`")
                            break
                if referenced:
                    break  

        if not referenced:
            print(f"Skipping `{table_name}`, no valid references found in this snapshot.!!! ")
            continue  

        print(f" Adding `{table_name}` to snapshot with {len(df)} rows.")

        col_to_stype = col_to_stype_dict[table_name]
        col_to_stype_copy = col_to_stype.copy()
        remove_pkey_fkey(col_to_stype_copy, table)  

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize()

        snapshot[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

    print("\n=== Finished Processing Related Tables ===\n")
    return col_stats_dict

def add_edges(db, snapshot, ts) -> bool:
    has_edges = False
    for table_name, table in db.table_dict.items():
        # Only proceed if the table (as a node) exists in the snapshot.
        if table_name not in snapshot.node_types:
            continue

        df = table.df
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            # Only add edges if both the current table and the referenced table exist in the snapshot.
            if table_name not in snapshot.node_types or pkey_table_name not in snapshot.node_types:
                continue
            if fkey_name not in df.columns:
                continue

            pkey_column = db.table_dict[pkey_table_name].pkey_col
            if not pkey_column:
                print(f"⚠️ No primary key defined for {pkey_table_name}. Skipping edge creation.")
                continue

            # Extract primary key values from the snapshot.
            pkey_tensor_frame = snapshot[pkey_table_name].tf
            all_columns = sum(pkey_tensor_frame.col_names_dict.values(), [])
            if pkey_column not in all_columns:
                print(f"Primary key column '{pkey_column}' not found in {pkey_table_name}. Skipping edge.")
                continue

            for stype_key, col_list in pkey_tensor_frame.col_names_dict.items():
                if pkey_column in col_list:
                    stype_idx = col_list.index(pkey_column)
                    break
            else:
                print(f"Could not find stype key for '{pkey_column}'. Skipping edge.")
                continue

            valid_pkeys = pkey_tensor_frame.feat_dict[stype_key][:, stype_idx].numpy()
            df_filtered = df[df[fkey_name].isin(valid_pkeys)]
            if df_filtered.empty:
                continue

            fkey_index = torch.arange(len(df_filtered), dtype=torch.long)
            pkey_index = torch.tensor(df_filtered[fkey_name].values, dtype=torch.long)
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            snapshot[edge_type].edge_index = sort_edge_index(edge_index)

            ts_numeric = ts.timestamp()
            snapshot[edge_type].edge_time = torch.full((edge_index.size(1),), ts_numeric, dtype=torch.float32)
            has_edges = True

    return has_edges


def ensure_nodes_from_edges(snapshot: HeteroData, feature_dim=128):
    """
    Ensure that every node referenced in edges has a valid feature matrix 'x'
    of shape (num_nodes, feature_dim). If not, create or extend it with zeros.
    """
    for edge_type in snapshot.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = snapshot[edge_type].edge_index

        for node_type, nodes in [(src_type, edge_index[0]), (dst_type, edge_index[1])]:
            # If the node type doesn't exist, initialize it.
            if node_type not in snapshot.node_types:
                snapshot[node_type] = HeteroData()
                snapshot[node_type].num_nodes = 0

            current_num_nodes = getattr(snapshot[node_type], 'num_nodes', 0)
            max_index = nodes.max().item() if nodes.numel() > 0 else -1
            if max_index >= current_num_nodes:
                snapshot[node_type].num_nodes = max_index + 1

            # Ensure there is a feature matrix 'x'
            if not hasattr(snapshot[node_type], 'x'):
                snapshot[node_type].x = torch.zeros((snapshot[node_type].num_nodes, feature_dim))
            else:
                x = snapshot[node_type].x
                if x.size(0) < snapshot[node_type].num_nodes:
                    additional = torch.zeros((snapshot[node_type].num_nodes - x.size(0), x.size(1)), device=x.device)
                    snapshot[node_type].x = torch.cat([x, additional], dim=0)


def make_snapshot_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    main_table_name: str,
    interval_days: int,
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[HeteroData], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Construct a sequence of graph snapshots at fixed timestamps using a main table 
       and a defined interval in days for snapshot creation."""

    data_snapshots = []
    col_stats_dict = dict()

    # Validate main table
    if main_table_name not in db.table_dict:
        raise ValueError(f"Main table {main_table_name} not found in database.")

    main_table = db.table_dict[main_table_name]
    if main_table.time_col is None or main_table.time_col not in main_table.df.columns:
        raise ValueError(f"Main table {main_table_name} must have a valid timestamp column.")

    # Generate timestamps based on the main table
    timestamps = generate_timestamps(main_table, interval_days)

    print("******* Time Stamps:", len(timestamps))
    if not timestamps:
        return [], {}

    # Create snapshots at defined timestamps
    for ts in timestamps:
        snapshot = HeteroData()

        # Process main tables (filter by timestamp)
        main_col_stats = process_main_tables(db, col_to_stype_dict, snapshot, ts, text_embedder_cfg)
        col_stats_dict.update(main_col_stats)

        # Process related FK tables
        related_col_stats = process_related_tables(db, col_to_stype_dict, snapshot, text_embedder_cfg)
        col_stats_dict.update(related_col_stats)

        # Add edges ensuring FK relationships are consistent with the snapshot
        has_edges = add_edges(db, snapshot, ts)
        
        # Ensure all nodes from edges are present
        ensure_nodes_from_edges(snapshot)
        
        # Ensure snapshot contains at least one edge
        if not has_edges:
            print(f" ****** Snapshot at timestamp {ts} has no edges. Skipping it.")
            continue

        snapshot.validate()
        data_snapshots.append(snapshot)

    return data_snapshots, col_stats_dict
    
class AttachTargetTransform:
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of disjoins subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch

class NodeTrainTableInput(NamedTuple):
    r"""Training table input for node prediction.

    - nodes is a Tensor of node indices.
    - time is a Tensor of node timestamps.
    - target is a Tensor of node labels.
    - transform attaches the target to the batch.
    """

    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(
    table: Table,
    task: EntityTask,
) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


class LinkTrainTableInput(NamedTuple):
    r"""Training table input for link prediction.

    - src_nodes is a Tensor of source node indices.
    - dst_nodes is PyTorch sparse tensor in csr format.
        dst_nodes[src_node_idx] gives a tensor of destination node
        indices for src_node_idx.
    - num_dst_nodes is the total number of destination nodes.
        (used to perform negative sampling).
    - src_time is a Tensor of time for src_nodes
    """

    src_nodes: Tuple[NodeType, Tensor]
    dst_nodes: Tuple[NodeType, Tensor]
    num_dst_nodes: int
    src_time: Optional[Tensor]


def get_link_train_table_input(
    table: Table,
    task: RecommendationTask,
) -> LinkTrainTableInput:
    r"""Get the training table input for link prediction."""

    src_node_idx: Tensor = torch.from_numpy(
        table.df[task.src_entity_col].astype(int).values
    )
    exploded = table.df[task.dst_entity_col].explode()
    coo_indices = torch.from_numpy(
        np.stack([exploded.index.values, exploded.values.astype(int)])
    )
    sparse_coo = torch.sparse_coo_tensor(
        coo_indices,
        torch.ones(coo_indices.size(1), dtype=bool),
        (len(src_node_idx), task.num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    return LinkTrainTableInput(
        src_nodes=(task.src_entity_table, src_node_idx),
        dst_nodes=(task.dst_entity_table, dst_node_indices),
        num_dst_nodes=task.num_dst_nodes,
        src_time=time,
    )
