import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
THRESHOLD = 1.5
MIN_INTERACTIONS = 5
dataset_path = './yoochoose-dataset'

def read_yoochoose_data():
    clicks = pd.read_csv(
        os.path.join(dataset_path, 'yoochoose-clicks.dat'),
        names=["session", "timestamp", "item", "category"],
        parse_dates=["timestamp"],
        dtype={"category": str}
    )
    clicks['rating'] = 1

    buys = pd.read_csv(
        os.path.join(dataset_path, 'yoochoose-buys.dat'),
        names=["session", "timestamp", "item", "price", "qty"],
        parse_dates=["timestamp"]
    )
    buys['rating'] = 3

    data = pd.concat([clicks[['session', 'item', 'rating']], buys[['session', 'item', 'rating']]])
    data.rename(columns={'session': 'visitorid', 'item': 'itemid'}, inplace=True)
    return data

def filter_users(data, min_interactions=MIN_INTERACTIONS):
    user_counts = data['visitorid'].value_counts()
    users_to_keep = user_counts[user_counts >= min_interactions].index
    return data[data['visitorid'].isin(users_to_keep)]

def split_train_test_vectorized(data, test_prop=0.2):
    np.random.seed(SEED)
    test_mask = np.random.rand(len(data)) < test_prop
    return data[~test_mask], data[test_mask]

raw_data = read_yoochoose_data()
transformed_data = filter_users(raw_data)

# Crear mapeos de usuarios e ítems
unique_users = transformed_data['visitorid'].unique()
unique_items = transformed_data['itemid'].unique()

user2id = {uid: i for i, uid in enumerate(unique_users)}
item2id = {iid: i for i, iid in enumerate(unique_items)}

transformed_data['uid'] = transformed_data['visitorid'].map(user2id)
transformed_data['sid'] = transformed_data['itemid'].map(item2id)

train_set, temp_set = train_test_split(transformed_data, test_size=0.2, random_state=SEED)
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=SEED)

val_data_tr, val_data_te = split_train_test_vectorized(val_set)
test_data_tr, test_data_te = split_train_test_vectorized(test_set)

output_dir = './processed_yoochoose'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_set.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_data_tr.to_csv(os.path.join(output_dir, 'validation_tr.csv'), index=False)
val_data_te.to_csv(os.path.join(output_dir, 'validation_te.csv'), index=False)
test_data_tr.to_csv(os.path.join(output_dir, 'test_tr.csv'), index=False)
test_data_te.to_csv(os.path.join(output_dir, 'test_te.csv'), index=False)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for item in unique_items:
        f.write(f'{item}\n')

with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for user in unique_users:
        f.write(f'{user}\n')

print("Preprocesamiento completado y archivos guardados en el directorio 'processed_yoochoose'.")
