"""Train TS2Vec encoder on preprocessed DNA dataset and output embeddings.

Reads:  data/dataset.npy — shape (N, 200, 4)
Writes: models/ts2vec.pkl — trained model checkpoint
        data/embeddings.npy — shape (N, 320) instance-level
        data/trajectory_embeddings.npy — shape (N, 200, 320) timestamp-level
"""

import os
import sys
import time

import numpy as np

DATA_DIR = "data"
MODEL_DIR = "models"


def main():
    from ts2vec import TS2Vec

    # Load dataset
    dataset_path = os.path.join(DATA_DIR, "dataset.npy")
    if not os.path.isfile(dataset_path):
        print("ERROR: data/dataset.npy not found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    data = np.load(dataset_path)
    n_samples, seq_len, n_features = data.shape
    print(f"Loaded dataset: {data.shape} ({n_samples} tokens, {seq_len} steps, {n_features} dims)")

    # Initialize model
    model = TS2Vec(
        input_dims=n_features,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cpu",
        lr=0.001,
        batch_size=min(16, n_samples),
    )

    # Train
    print("Training TS2Vec...")
    t0 = time.time()
    loss_log = model.fit(data, n_epochs=200, verbose=True)
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s, final loss: {loss_log[-1]:.6f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "ts2vec.pkl")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    # Encode — instance-level (for point-based clustering)
    print("Encoding instance-level embeddings...")
    embeddings = model.encode(data, encoding_window="full_series")
    print(f"Instance embeddings shape: {embeddings.shape}")
    emb_path = os.path.join(DATA_DIR, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Instance embeddings saved: {emb_path}")

    # Encode — timestamp-level (for trajectory visualization)
    print("Encoding timestamp-level embeddings...")
    traj_embeddings = model.encode(data)
    print(f"Trajectory embeddings shape: {traj_embeddings.shape}")
    traj_path = os.path.join(DATA_DIR, "trajectory_embeddings.npy")
    np.save(traj_path, traj_embeddings)
    print(f"Trajectory embeddings saved: {traj_path}")


if __name__ == "__main__":
    main()
