import argparse
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import sys


# Hyperparameters and paths via argparse
parser = argparse.ArgumentParser(description="Train a protein classifier")
parser.add_argument("--train_dataset_path", default="./dataset/train_dataset.csv", type=str, help="Path to training dataset")
parser.add_argument("--valid_dataset_path", default="./dataset/valid_dataset.csv", type=str, help="Path to validation dataset")
parser.add_argument("--test_dataset_path", default="./dataset/test_dataset.csv", type=str, help="Path to test dataset")
parser.add_argument("--num_epochs", default=400, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training and validation")
parser.add_argument("--w2v_embedding_size", default=21, type=int, help="Embedding size for Word2Vec")
parser.add_argument("--w2v_num_epochs", default=10, type=int, help="Number of epochs for Word2Vec")
parser.add_argument("--embedding_size", default=1024, type=int, help="Embedding size for fc layers")
parser.add_argument("--dropout", type=float, default=0.6, help="Dropout probability")
parser.add_argument("--num_shots", default=None, type=int, help="Number of samples for few-shot learning. If not provided, the full dataset is used.")
args = parser.parse_args()

# Initialize constants and dictionary
quant_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def quantize_sequence(sequence):
    return [quant_dict[aa] for aa in sequence if aa in quant_dict]

def to_three_grams(sequence):
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]

train_dataset_path = args.train_dataset_path
valid_dataset_path = args.valid_dataset_path
test_dataset_path = args.test_dataset_path

train_dataset = pd.read_csv(train_dataset_path)[['Protein families', 'Sequence']]
valid_dataset = pd.read_csv(valid_dataset_path)[['Protein families', 'Sequence']]
test_dataset = pd.read_csv(test_dataset_path)[['Protein families', 'Sequence']]

if args.num_shots:
    train_dataset = train_dataset.sample(n=args.num_shots, replace=False)
    valid_dataset = valid_dataset.sample(n=args.num_shots, replace=False)

train_dataset.columns = ['Label', 'Seq']
valid_dataset.columns = ['Label', 'Seq']
test_dataset.columns = ['Label', 'Seq']

def preprocess_data(dataset):
    dataset.drop_duplicates(subset='Seq').dropna(subset=['Seq', 'Label'])
    dataset['QuantizedSeq'] = dataset['Seq'].apply(quantize_sequence)
    dataset['ThreeGrams'] = dataset['Seq'].apply(to_three_grams)
    return dataset

# Preprocess datasets
train_dataset = preprocess_data(train_dataset)
valid_dataset = preprocess_data(valid_dataset)
test_dataset = preprocess_data(test_dataset)

# Combine 'ThreeGrams' for Word2Vec
all_three_grams = train_dataset['ThreeGrams'].tolist() + valid_dataset['ThreeGrams'].tolist()

# Word2Vec embedding
embedding_size = args.w2v_embedding_size
w2v_model = Word2Vec(vector_size=embedding_size, window=5, min_count=1, workers=16, alpha=0.025, min_alpha=0.0001)
w2v_model.build_vocab(all_three_grams)
w2v_model.train(all_three_grams, total_examples=len(all_three_grams), epochs=args.w2v_num_epochs)

def sequence_to_embedding(seq):
    embeddings = [w2v_model.wv[three_gram] for three_gram in seq if three_gram in w2v_model.wv]
    return embeddings

def normalize_embeddings(embeddings):
    max_val = max([max(e) for e in embeddings])
    min_val = min([min(e) for e in embeddings])
    return [(e - min_val) / (max_val - min_val) for e in embeddings]

def get_normalized_embedding(seq):
    embeddings = sequence_to_embedding(seq)
    return normalize_embeddings(embeddings)

# Directly create 'NormalizedEmbeddings' without intermediate 'Embeddings'
train_dataset['NormalizedEmbeddings'] = train_dataset['ThreeGrams'].apply(get_normalized_embedding)
valid_dataset['NormalizedEmbeddings'] = valid_dataset['ThreeGrams'].apply(get_normalized_embedding)
test_dataset['NormalizedEmbeddings'] = test_dataset['ThreeGrams'].apply(get_normalized_embedding)

# Drop the 'ThreeGrams' column as it's no longer needed
train_dataset.drop(columns=['ThreeGrams'], inplace=True)
valid_dataset.drop(columns=['ThreeGrams'], inplace=True)
test_dataset.drop(columns=['ThreeGrams'], inplace=True)

num_classes = train_dataset['Label'].nunique()
unique_labels = train_dataset['Label'].unique().tolist()
label_map = {label: idx for idx, label in enumerate(unique_labels)}

class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length=None):
        self.data = dataframe
        # If max_length is not provided, use the maximum sequence length in the dataframe
        self.max_length = max_length or max(dataframe['NormalizedEmbeddings'].apply(len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['NormalizedEmbeddings']
        # Convert sequence to a numpy array first
        sequence_array = np.array(sequence, dtype=np.float32)
        # Convert the numpy array to a tensor
        sequence_tensor = torch.tensor(sequence_array)
        padded_tensor = torch.zeros(self.max_length, sequence_tensor.shape[1])
        padded_tensor[:sequence_tensor.shape[0], :] = sequence_tensor
        label = torch.tensor(self.data.iloc[idx]['Label'], dtype=torch.long)
        return padded_tensor, label

class ProteinFamilyClassifier(nn.Module):
    def __init__(self):
        super(ProteinFamilyClassifier, self).__init__()

        # GRU layer (Replacing the CNN layers)
        self.gru = nn.GRU(input_size=args.w2v_embedding_size, hidden_size=512, num_layers=2, dropout=args.dropout, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2, args.embedding_size)  # 512 * 2 because it's bidirectional
        self.fc2 = nn.Linear(args.embedding_size, args.embedding_size)
        self.fc3 = nn.Linear(args.embedding_size, num_classes)  # num_classes: number of protein families

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # x shape: batch_size, seq_len, input_size
        h0 = torch.zeros(2 * 2, x.size(0), 512).to(device)  # 2 for bidirection, 2 for number of layers
        _, h_n = self.gru(x, h0)  # GRU outputs: output, h_n
        # h_n shape: num_layers * num_directions, batch, hidden_size. We're interested in the last hidden state
        x = h_n[-2:].transpose(0, 1).contiguous().view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Create datasets and data loaders
batch_size = args.batch_size
train_dataset_obj = ProteinDataset(train_dataset)
valid_dataset_obj = ProteinDataset(valid_dataset)
test_dataset_obj = ProteinDataset(test_dataset)

train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset_obj, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_obj, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinFamilyClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

num_epochs = args.num_epochs  # example

writer = SummaryWriter()
print(f"Training started")
sys.stdout.flush()
for epoch in range(args.num_epochs):
    model.train()
    # Training Loop without tqdm
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation Loop without tqdm
    model.eval()
    valid_loss = 0.0
    all_predictions = []
    all_true_labels = []

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    recall = recall_score(all_true_labels, all_predictions, average='macro')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        precision = precision_score(all_true_labels, all_predictions, average='macro')

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {train_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    sys.stdout.flush()
    # Log metrics to TensorBoard
    writer.add_scalar('Training Loss', train_loss/len(train_loader), epoch)
    writer.add_scalar('Validation Loss', valid_loss/len(valid_loader), epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    writer.add_scalar('Validation F1 Score', f1, epoch)
    writer.add_scalar('Validation Recall', recall, epoch)
    writer.add_scalar('Validation Precision', precision, epoch)

    # Test Loop without tqdm
    test_loss = 0.0
    all_test_predictions = []
    all_test_true_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_test_predictions.extend(predicted.cpu().numpy())
        all_test_true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_test_true_labels, all_test_predictions)
    test_f1 = f1_score(all_test_true_labels, all_test_predictions, average='macro')
    test_recall = recall_score(all_test_true_labels, all_test_predictions, average='macro')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        test_precision = precision_score(all_test_true_labels, all_test_predictions, average='macro')

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}")
    print("--------------------------------------------------------------------------------")
    sys.stdout.flush()

    # Log test metrics to TensorBoard
    writer.add_scalar('Test Loss', test_loss/len(test_loader), epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    writer.add_scalar('Test F1 Score', test_f1, epoch)
    writer.add_scalar('Test Recall', test_recall, epoch)
    writer.add_scalar('Test Precision', test_precision, epoch)

print(f"Training finished")
sys.stdout.flush()
writer.close()
