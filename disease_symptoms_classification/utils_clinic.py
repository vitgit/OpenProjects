import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import pandas as pd

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def df_token_counts(df, limit):
    av_tokens_in_text = 0.0
    max_tokens = 0
    min_tokens = 100000
    num_more_than_limit = 0
    for index, row in df.iterrows():
        text = row['text']
        num_tokens = count_tokens(text)
        av_tokens_in_text += num_tokens
        if num_tokens > max_tokens:
            max_tokens = num_tokens
        if num_tokens < min_tokens:
            min_tokens = num_tokens
        if num_tokens > limit:
            num_more_than_limit += 1
    av_tokens_in_text = round(float(av_tokens_in_text / len(df)), 2)
    num_more_than_limit = round(float(num_more_than_limit / len(df) * 100), 2)
    print('\nav_tokens_in_text:', av_tokens_in_text, '  max_tokens:', max_tokens,
          '  min_tokens:', min_tokens, '   num_more_than_limit:', num_more_than_limit, '%\n')

def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)

def create_folder(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)
    print(dir, 'has been created or checked')

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
def empty_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
def exclude_labels_with_small_occurrence(df1, df2=None, limit=1, label_map=None):
    # Group by 'label' and count the occurrences in df1
    label_counts = df1.groupby('label').size().reset_index(name='count')

    # Identify labels that have occurrences less than or equal to the limit
    labels_with_fewer_docs = label_counts[label_counts['count'] <= limit]['label']
    labels_with_fewer_docs_list = labels_with_fewer_docs.tolist()

    print(f'labels_with_fewer_docs: {labels_with_fewer_docs_list}')
    print(f'Number of labels_with_fewer_docs: {len(labels_with_fewer_docs_list)}')

    # If df2 is None, set it to df1
    if df2 is None:
        df2 = df1

    # Exclude rows with labels that have only few documents from df2
    filtered_df = df2[~df2['label'].isin(labels_with_fewer_docs_list)]

    # Filter the label_map if provided
    filtered_label_map = {k: v for k, v in label_map.items() if v not in labels_with_fewer_docs_list} if label_map else None

    return filtered_df, filtered_label_map

def stratified_split_with_min_representations(df, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    train_list = []
    test_list = []

    # Group by label and process each group
    for label, group in df.groupby('label'):
        n_samples = len(group)
        n_test = max(1, int(n_samples * test_size))

        if n_samples > 1:
            # Randomly shuffle the group
            shuffled_group = group.sample(frac=1, random_state=random_state)
            test_indices = shuffled_group.index[:n_test]
            train_indices = shuffled_group.index[n_test:]
        else:
            # If there is only one sample, ensure it goes to the training set
            test_indices = []
            train_indices = group.index

        test_list.extend(test_indices)
        train_list.extend(train_indices)

    train_df = df.loc[train_list]
    test_df = df.loc[test_list]

    # Count labels in the training set
    label_counts = train_df['label'].value_counts().to_dict()
    sorted_label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))

    # Get max and min frequencies
    max_frequency = next(iter(sorted_label_counts.values())) if sorted_label_counts else None
    min_frequency = next(iter(reversed(sorted_label_counts.values()))) if sorted_label_counts else None

    return train_df, test_df, sorted_label_counts, max_frequency, min_frequency

def re_label_dataframes(train_df, test_df, label_map):
    # Create a new label map with continuous values starting from 0
    unique_labels = sorted(train_df['label'].unique())
    new_label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Apply the new label map to train_df and test_df
    train_df['label']   = train_df['label'].map(new_label_map)
    test_df['label']    = test_df['label'].map(new_label_map)

    # Update the original label_map
    updated_label_map = {key: new_label_map[value] for key, value in label_map.items() if value in new_label_map}

    return train_df, test_df, updated_label_map

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
def plot_overfitting(train_losses, val_losses, train_accuracies, val_accuracies, file):
    pdf = matplotlib.backends.backend_pdf.PdfPages(file)

    epochs = range(1, len(train_losses) + 1)

    fig = plt.figure(figsize=(14, 5))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    pdf.close()

def organize_data(train_df1, test_df1, label_map_orig, test_size):
    # Combine
    combined_df = pd.concat([train_df1, test_df1], ignore_index=True)

    # replace "_"
    def replace_underscores(text):
        return text.replace("_", " ")
    combined_df['text'] = combined_df['text'].apply(replace_underscores)

    # remove duplicates
    combined_df = combined_df.drop_duplicates(subset='text')

    # Filter
    df, label_map = exclude_labels_with_small_occurrence(combined_df, df2=None, limit=1, label_map=label_map_orig)
    train_df, test_df, sorted_label_counts, max_frequency, min_frequency = \
        stratified_split_with_min_representations(df, test_size=test_size, random_state=42)
    print(f'labels counts distribution in training set:\n{sorted_label_counts}')
    print(f'Total labels in training set: {len(sorted_label_counts)}')
    print(f'Max frequency: {max_frequency}')
    print(f'Min frequency: {min_frequency}')

    train_df, test_df, label_map = re_label_dataframes(train_df, test_df, label_map)
    print(label_map)
    print(train_df)
    print(test_df)
    return train_df, test_df, label_map
def clean_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)