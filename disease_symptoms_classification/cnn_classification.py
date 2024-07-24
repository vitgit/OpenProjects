# https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer

from utils_clinic import empty_dir, save_checkpoint, set_seed, create_folder, \
    df_token_counts, create_data_loader, plot_overfitting, organize_data

device = torch.device("cpu")
loss_fn = nn.CrossEntropyLoss().to(device)

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits

def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01):
    """Instantiate a CNN model and an optimizer."""

    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout)

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95)

    return cnn_model, optimizer
def train(model, optimizer, train_dataloader, val_dataloader, checkpoint_file, start_training, fig_file, epochs=10):
    """Train the CNN model."""

    if start_training:
        dir = os.path.dirname(checkpoint_file)
        empty_dir(dir)

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    train_losses        = []
    val_losses          = []
    train_accuracies    = []
    val_accuracies      = []

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        train_accuracy = []
        all_labels = []
        all_preds = []

        for step, d in enumerate(train_dataloader):
            b_input_ids = d["input_ids"].to(device)
            b_labels = d["label"].to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)
            preds = torch.argmax(logits, dim=1).flatten()

            all_labels.extend(b_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            train_accuracy.append(accuracy)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        train_accuracy_av = np.mean(train_accuracy)
        train_accuracy_score = accuracy_score(all_labels, all_preds)

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy_av)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy, accuracy, precision, recall, f1 = evaluate(model, val_dataloader)

            print(f'Val Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

        # Save checkpoint
        ch_file = checkpoint_file.replace('@@@@', str(epoch_i + 1))
        save_checkpoint({
                'epoch': epoch_i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
        }, filename=ch_file)


    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    plot_overfitting(train_losses, val_losses, train_accuracies, val_accuracies, fig_file)


def evaluate(model, val_dataloader):

    # Put the model into the evaluation mode. The dropout layers are disabled
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    all_labels = []
    all_preds = []

    for d in val_dataloader:
        # text = d["text"]
        b_input_ids = d["input_ids"].to(device)
        b_labels = d["label"].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        all_labels.extend(b_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

    return val_loss, val_accuracy, accuracy, precision, recall, f1

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    set_seed(42)
    #======================================================
    train_data_file     = base_dir + '/diseases_symptoms/symptom-disease-train-dataset.csv'
    test_data_file      = base_dir + '/diseases_symptoms/symptom-disease-test-dataset.csv'
    mapping_file        = base_dir + '/diseases_symptoms/mapping.json'

    # train_data_file     = base_dir + '/Medical-Abstracts-TC-Corpus/medical_tc_train.csv'
    # test_data_file      = base_dir + '/Medical-Abstracts-TC-Corpus/medical_tc_test.csv'
    # mapping_file        = base_dir + '/Medical-Abstracts-TC-Corpus/mapping.json'

    checkpoint_dir      = 'C:/fine_tune_models/CNN_test'

    create_folder(checkpoint_dir)
    checkpoint_file     = checkpoint_dir + '/' + '@@@@' + '_checkpoint.pth.tar'

    fig_file            = base_dir + '/plots/overfitting_cnn.pdf'

    EMBEDDING_DIM       = 300
    KERNEL_SIZES        = [3, 4, 5]
    NUM_FILTERS         = [100, 100, 100]
    DROPOUT             = 0.5
    '''
    EMBEDDING_DIM   - The embedding dimension determines the size of the word vectors.
    KERNEL_SIZES    - The kernel sizes define the sizes of the convolutional filters used to detect patterns of different lengths.
    NUM_FILTERS     - The number of filters specifies how many patterns the model will learn for each kernel size.
    DROPOUT         - The dropout rate controls the regularization to prevent overfitting.
    '''
    EPOCHS              = 35
    batch_size          = 8
    max_len             = 420
    learning_rate       = 0.05

    test_size           = 0.1

    start_training      = True
    #=======================================================

    # Load data
    train_df1 = pd.read_csv(train_data_file)
    test_df1 = pd.read_csv(test_data_file)

    # Load label mapping
    with open(mapping_file) as f:
        label_map_orig = json.load(f)

    print(f'Original labels count: {len(label_map_orig)}')

    train_df, test_df, label_map = organize_data(train_df1, test_df1, label_map_orig, test_size)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    VOCAB_SIZE = tokenizer.vocab_size
    print('\nVOCAB_SIZE:', VOCAB_SIZE, '\n')
    NUM_CLASSES = len(label_map)
    print('\nNUM_CLASSES:', NUM_CLASSES, '\n')

    df_token_counts(train_df, max_len)
    df_token_counts(test_df, max_len)

    # Load data loaders
    train_data_loader = create_data_loader(train_df, tokenizer, max_len=max_len, batch_size=batch_size)
    test_data_loader = create_data_loader(test_df, tokenizer, max_len=max_len, batch_size=batch_size)

    cnn_rand, optimizer = initilize_model(
                                            pretrained_embedding=None,
                                            freeze_embedding=False,
                                            vocab_size=VOCAB_SIZE,
                                            embed_dim=EMBEDDING_DIM,
                                            filter_sizes=KERNEL_SIZES,
                                            num_filters=NUM_FILTERS,
                                            num_classes=NUM_CLASSES,
                                            dropout=DROPOUT,
                                            learning_rate=learning_rate
                                         )
    train(cnn_rand, optimizer, train_data_loader, test_data_loader, checkpoint_file, start_training, fig_file, epochs=EPOCHS)