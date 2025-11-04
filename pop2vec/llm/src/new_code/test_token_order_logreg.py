import argparse
import logging
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader

from pop2vec.llm.src.new_code.load_data import CustomLazyHDF5Dataset
from pop2vec.llm.src.transformer.models import TransformerEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, no_age_emb=False, no_date_emb=False):
    """Load pretrained model with optional positional embedding exclusion."""
    logger.info(f"Loading model from {checkpoint_path}")
    logger.info(f"Positional embedding exclusion: age={no_age_emb}, date={no_date_emb}")
    
    model = TransformerEncoder.load_from_checkpoint(
        checkpoint_path,
        no_age_emb=no_age_emb,
        no_date_emb=no_date_emb,
    )
    model = model.transformer
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded and moved to {device}")
    
    return model


def generate_token_pairs_from_sequence(sequence_tokens, num_pairs=5):
    """
    Generate token pairs from a sequence with order labels.
    
    Args:
        sequence_tokens: 1D array/tensor of token IDs (from input_ids[:, 0])
        num_pairs: Number of pairs to generate per sequence
    
    Returns:
        pairs: List of (pos1, pos2) tuples (positions in sequence)
        labels: List of labels (1 if pos1 < pos2, else 0)
    """
    # Convert to numpy and remove padding (assuming 0 is padding)
    if torch.is_tensor(sequence_tokens):
        sequence_tokens = sequence_tokens.cpu().numpy()
    
    non_padding_mask = sequence_tokens != 0
    valid_positions = np.where(non_padding_mask)[0]
    
    if len(valid_positions) < 2:
        return [], []
    
    pairs = []
    labels = []
    
    # Generate num_pairs random pairs
    for _ in range(num_pairs):
        # Sample two distinct positions
        sampled_positions = np.random.choice(valid_positions, size=2, replace=False)
        pos1, pos2 = sampled_positions[0], sampled_positions[1]
        
        # Create both ordered and reversed pairs
        # Ordered pair (label=1 if pos1 < pos2, else 0)
        if pos1 < pos2:
            pairs.append((pos1, pos2))
            labels.append(1)
            # Also add reversed
            pairs.append((pos2, pos1))
            labels.append(0)
        else:
            pairs.append((pos1, pos2))
            labels.append(0)
            # Also add reversed
            pairs.append((pos2, pos1))
            labels.append(1)
    
    return pairs, labels


def get_token_embeddings_at_positions(model, input_ids, padding_mask, positions):
    """
    Get embeddings for specific token positions in a sequence.
    
    Args:
        model: Transformer model
        input_ids: [1, 4, seq_len] (batch size is 1)
        padding_mask: [1, seq_len]
        positions: List of positions to extract
    
    Returns:
        embeddings: [num_positions, hidden_dim]
    """
    with torch.no_grad():
        # Get full sequence embeddings
        outputs = model(x=input_ids, padding_mask=padding_mask)  # [1, seq_len, hidden_dim]
        # logger.info(f"Model output shape: {outputs.shape}")

        # Extract embeddings at specific positions
        embeddings = outputs[0, positions, :]  # [num_positions, hidden_dim]
        # logger.info(f"Extracted embeddings shape: {embeddings.shape}")

    return embeddings.cpu().numpy()


def prepare_dataset(tokenized_path, model, num_sequences=100, pairs_per_seq=5, 
                    zero_age=False, zero_date=False):
    """
    Prepare dataset for token order prediction using real HDF5 data.
    
    Args:
        tokenized_path: Path to tokenized sequences HDF5
        model: Pretrained model
        num_sequences: Number of sequences to sample
        pairs_per_seq: Number of token pairs per sequence
        zero_age: Whether to zero out age information
        zero_date: Whether to zero out date information
    
    Returns:
        X: Feature matrix [num_samples, feature_dim]
        y: Labels [num_samples]
    """
    X_list = []
    y_list = []
    
    # Load dataset using the same approach as infer_embedding.py
    dataset = CustomLazyHDF5Dataset(
        tokenized_path,
        validation=False,
        inference=True,
        mlm_encoded=False,
        num_val_items=0,
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} sequences")
    
    # Sample indices
    total_sequences = len(dataset)
    sample_size = min(num_sequences, total_sequences)
    sample_indices = np.random.choice(total_sequences, size=sample_size, replace=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for idx in tqdm(sample_indices, desc="Processing sequences"):
        # Get sample from dataset
        sample = dataset[idx]
        logger.info(f"Processing sequence index: {idx}")

        input_ids = sample["input_ids"].unsqueeze(0)  # [1, 4, seq_len]
        padding_mask = sample["padding_mask"].unsqueeze(0)  # [1, seq_len]
        
        # Zero out age/date if requested
        if zero_age:
            input_ids[:, 2, :] = 0  # Age is at index 2
        if zero_date:
            input_ids[:, 1, :] = 0  # Date is at index 1
        
        # Move to device
        input_ids = input_ids.to(device)
        padding_mask = padding_mask.to(device)
        
        # Generate token pairs based on positions
        sequence_tokens = input_ids[0, 0, :]  # Token IDs (dimension 0)
        # logger.info(f"Length of sequence tokens: {sequence_tokens.shape}")

        pairs, labels = generate_token_pairs_from_sequence(sequence_tokens, num_pairs=pairs_per_seq)
        
        if not pairs:
            continue
        
        for (pos1, pos2), label in zip(pairs, labels):
            # Get embeddings for the two token positions
            embeddings = get_token_embeddings_at_positions(model, input_ids, padding_mask, [pos1, pos2])
            
            # Feature: concatenate the two token embeddings
            feature = np.concatenate([embeddings[0], embeddings[1]])
            
            X_list.append(feature)
            y_list.append(label)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"Generated {len(X)} samples with feature dim {X.shape[1]}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Test token order prediction with logistic regression")
    parser.add_argument("--tokenized_path", required=True, help="Path to tokenized sequences HDF5")
    parser.add_argument("--checkpoint_path", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--num_sequences", type=int, default=100, help="Number of sequences to sample")
    parser.add_argument("--pairs_per_seq", type=int, default=5, help="Number of token pairs per sequence")
    parser.add_argument("--no_age_emb", action="store_true", help="Exclude age positional embeddings")
    parser.add_argument("--no_date_emb", action="store_true", help="Exclude date positional embeddings")
    parser.add_argument("--zero_age", action="store_true", help="Zero out age input")
    parser.add_argument("--zero_date", action="store_true", help="Zero out date input")
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint_path, no_age_emb=args.no_age_emb, no_date_emb=args.no_date_emb)
    
    # Prepare dataset
    logger.info("Preparing dataset from real HDF5 data...")
    X, y = prepare_dataset(
        args.tokenized_path, 
        model, 
        num_sequences=args.num_sequences,
        pairs_per_seq=args.pairs_per_seq,
        zero_age=args.zero_age,
        zero_date=args.zero_date
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train logistic regression
    logger.info("Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    confusion_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Results:")
    logger.info(f"Confusion Matrix:\n{confusion_matrix}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"{'='*50}\n")
    logger.info("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Reversed (0)", "Ordered (1)"]))


if __name__ == "__main__":
    main()