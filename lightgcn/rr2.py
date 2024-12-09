import torch
import pandas as pd
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.model_selection import train_test_split

# Read df from CSV
def RETAILROCKET():
    df_train = pd.read_csv(r"processed_RetailRocket\train.csv")
    #df_train = df_train.sample(frac=0.1).reset_index(drop=True)
    # Create the heterogeneous dataset
    hetero_data = HeteroData()

    # Set the number of nodes for each type
    hetero_data['user'].num_nodes = len(df_train['uid'].unique())
    hetero_data['book'].num_nodes = len(df_train['sid'].unique())

    user_indices = {val: idx for idx, val in enumerate(df_train['uid'].unique())}
    book_indices = {val: idx for idx, val in enumerate(df_train['sid'].unique())}

    df_train["uid"] = df_train["uid"].map(user_indices)
    df_train["sid"] = df_train["sid"].map(book_indices)

    print(hetero_data)
    
    # Get the number of user and book nodes
    num_users = hetero_data['user'].num_nodes
    num_books = hetero_data['book'].num_nodes

    edges = pd.DataFrame(df_train[["uid", "sid"]].values, columns=["uid", "sid"])
    #as a tensor
    edges = torch.tensor(edges.values, dtype=torch.int64).t().contiguous()

    # Create random edges with offsets
    # Offset the book indices by the number of user nodes
    hetero_data["user", "rates", "book"].edge_index = edges
    hetero_data["book", "rated_by", "user"].edge_index = edges.flip(0)
    print(hetero_data)

    df_test = pd.read_csv(r"processed_yoochoose\test_te.csv")
    #sample
    #df_test = df_test.sample(frac=0.1).reset_index(drop=True)
    test_edges = pd.DataFrame(df_test[["uid", "sid"]].values, columns=["uid", "sid"])
    #sanity check that every user and book in test set is in train set
    test_edges = test_edges[test_edges["uid"].isin(user_indices.keys())]
    test_edges = test_edges[test_edges["sid"].isin(book_indices.keys())]

    test_edges["uid"] = test_edges["uid"].map(user_indices)
    test_edges["sid"] = test_edges["sid"].map(book_indices)
    test_edges = torch.tensor(test_edges.values, dtype=torch.int64).t().contiguous()
    hetero_data["user", "rates", "book"].edge_label_index = test_edges
    print(hetero_data)

    # df_test = pd.read_csv(r"YOUCHOOSE\raw\yoochoose-test.dat", delimiter=',', header=None)
    # df_test = df_test.sample(frac=0.1).reset_index(drop=True)
    # df_test.columns = ['session_id', 'timestamp', 'item_id', 'category']
    # #check that session id and item id are in session_id_map and item_id_map
    # df_test = df_test[df_test["session_id"].isin(session_id_map.keys())]
    # df_test = df_test[df_test["item_id"].isin(item_id_map.keys())]


    # test_user_indices = df_test["session_id"].map(session_id_map).values
    # test_book_indices = df_test["item_id"].map(item_id_map).values

    # #count number of nan
    # print(f"Number of nan in test_user_indices: {sum(pd.isnull(test_user_indices))}")
    # print(f"Number of nan in test_book_indices: {sum(pd.isnull(test_book_indices))}")

    # print(f"Max user index: {max(test_user_indices)}")
    # print(f"Max book index: {max(test_book_indices)}")

    #hetero_data["user", "rates", "book"].edge_label_index = torch.tensor([test_user_indices, test_book_indices + num_users], dtype=torch.int64).t().contiguous()
    # Print the HeteroData object to verify
    #print(hetero_data)
    data = hetero_data.to_homogeneous()
    print(data)
    return hetero_data

if __name__ == "__main__":
    YOUCHOOSE()
