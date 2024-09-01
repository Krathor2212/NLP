import pandas as pd


def load_and_transform_dataset(path):
    df = pd.read_csv(path, encoding='latin-1')
    df['Label'] = df['oh_label'].apply(lambda x: '1' if x == 0 else '0')
    data = df[['Text', 'Label']]
    data['Text'] = data['Text'].fillna('')
    return data

def load_dataset(file_paths):
    data_frames = [load_and_transform_dataset(file) for file in file_paths]
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data