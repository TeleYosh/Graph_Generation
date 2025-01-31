import random
import torch
from transformers import BertTokenizer, BertModel
import os 
from tqdm import tqdm

data_dir = 'data'

# Set a random seed
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def load_description(sample_id, dataset):
    """Load description for a given sample ID and dataset."""
    description_path = os.path.join(data_dir, dataset, 'description', f'graph_{sample_id}.txt')
    with open(description_path, 'r') as file:
        description = file.read().strip()
    return description

def compute_sentence_embeddings(descriptions):
    """
    Compute sentence embeddings for a list of descriptions using BERT.
    """
    # Tokenize and encode the entire batch
    encoding = tokenizer.batch_encode_plus(
        descriptions,
        padding=True,  # Pad all sequences to the same length
        truncation=True,  # Truncate sequences longer than the model's max length
        return_tensors='pt',  # Return PyTorch tensors
        add_special_tokens=True  # Add [CLS] and [SEP] tokens
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  # Token-level embeddings
        sentence_embeddings = word_embeddings.mean(dim=1)  # Average pooling over tokens

    return sentence_embeddings.numpy()  # Return as a NumPy array

def process_dataset_batch(data_dir, dataset_split):
    """
    Process dataset split (train/valid/test) and generate sentence embeddings.
    Handles the test set differently using a test.txt file.
    """
    descriptions = []
    ids = []

    if dataset_split == 'test':
        # Load test descriptions from test.txt
        test_file = os.path.join(data_dir, 'test', 'test.txt')
        print(f"Loading {dataset_split} descriptions from {test_file}...")
        with open(test_file, 'r') as file:
            for line in file:
                if line.strip():
                    graph_id, description = line.split(',', 1)
                    ids.append(graph_id.strip())
                    descriptions.append(description.strip())
    else:
        # Load train/valid descriptions from individual files
        sample_ids = os.listdir(os.path.join(data_dir, dataset_split, 'description'))
        print(f"Loading {dataset_split} descriptions...")
        for sample_id in tqdm(sample_ids, desc=f"Processing {dataset_split} descriptions"):
            sample_id = sample_id.split('.')[0].replace('graph_', '')  # Extract ID
            description = load_description(sample_id, dataset_split)
            descriptions.append(description)
            ids.append(sample_id)

    # Compute embeddings for all descriptions
    print(f"Generating embeddings for {dataset_split}...")
    sentence_embeddings = compute_sentence_embeddings(descriptions)
    shape = sentence_embeddings.shape

    # Create a dictionary mapping IDs to embeddings
    embeddings = {f"graph_{sample_id}": torch.tensor(embedding) for sample_id, embedding in zip(ids, sentence_embeddings)}

    return embeddings, shape

# Generate embeddings for train, valid, and test sets in a batch
train_embeddings, train_shape = process_dataset_batch(data_dir, 'train')
print('train_embeddings shape is: ', train_shape)
valid_embeddings, valid_shape = process_dataset_batch(data_dir, 'valid')
print('valid_embeddings shape is: ', valid_shape)
test_embeddings, test_shape = process_dataset_batch(data_dir, 'test')
print('test_embeddings shape is: ', test_shape)

# Save embeddings to disk
torch.save(train_embeddings, "train_embeddings.pt")
torch.save(valid_embeddings, "valid_embeddings.pt")
torch.save(test_embeddings, "test_embeddings.pt")