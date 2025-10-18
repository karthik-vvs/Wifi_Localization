"""
Improved FasterKAN WiFi Indoor Localization System
- Fixed WAP filtering (520 -> 465)
- GPU support optimized
- Better preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from typing import Dict, Tuple, List
import os

import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("✅ GPU is available")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("❌ No GPU available")

# ============================================================================
# DATASET UTILITIES
# ============================================================================

class WiFiDataset(Dataset):
    """Flexible WiFi dataset for regression and classification"""
    def __init__(self, X, y_reg=None, y_floor_bldg=None, y_space=None):
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg) if y_reg is not None else None
        self.y_floor_bldg = torch.LongTensor(y_floor_bldg) if y_floor_bldg is not None else None
        self.y_space = torch.LongTensor(y_space) if y_space is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx]}
        if self.y_reg is not None:
            item["y_reg"] = self.y_reg[idx]
        if self.y_floor_bldg is not None:
            item["y_floor_bldg"] = self.y_floor_bldg[idx]
        if self.y_space is not None:
            item["y_space"] = self.y_space[idx]
        return item

def load_and_preprocess_ujiindoor(path: str, task: str = "all"):
    """
    Load and preprocess UJIIndoorLoc dataset with FIXED WAP filtering
    """
    df = pd.read_csv(path)

    # Get WAP columns
    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    print(f"Original WAPs: {len(wap_cols)}")

    # CRITICAL FIX: Remove WAPs where ALL values are 100 (invalid)
    valid_waps = []
    for col in wap_cols:
        if not (df[col] == 100).all():  # Keep if at least one valid value
            valid_waps.append(col)

    print(f"Valid WAPs (removed all-100 columns): {len(valid_waps)}")

    # Extract features and replace 100 with -105
    X = df[valid_waps].values.astype(np.float32)
    X[X == 100] = -105

    results = {"X": X, "wap_cols": valid_waps, "n_features": len(valid_waps)}

    if task in ["regression", "all"]:
        coords = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_transformed = coords - coords_min

        results["coords"] = coords_transformed
        results["coords_min"] = coords_min
        results["coords_max"] = coords_max
        results["coords_range"] = coords_max - coords_min

        print(f"Coordinate ranges: {results['coords_range']}")

    if task in ["floor_building", "all"]:
        floor_bldg = df["FLOOR"].astype(str) + "_" + df["BUILDINGID"].astype(str)
        le_fb = LabelEncoder()
        floor_bldg_encoded = le_fb.fit_transform(floor_bldg)
        results["floor_bldg"] = floor_bldg_encoded
        results["floor_bldg_encoder"] = le_fb
        results["n_floor_bldg"] = len(le_fb.classes_)

    if task in ["space_id", "all"]:
        has_spaceid = df["SPACEID"].notna()

        if not has_spaceid.all():
            print(f"\nFiltering {(~has_spaceid).sum()} samples without SPACEID")
            X = X[has_spaceid]
            results["X"] = X

            if "coords" in results:
                results["coords"] = results["coords"][has_spaceid]
            if "floor_bldg" in results:
                results["floor_bldg"] = results["floor_bldg"][has_spaceid]

        space_ids = df.loc[has_spaceid, "SPACEID"].values
        le_space = LabelEncoder()
        space_encoded = le_space.fit_transform(space_ids)
        results["space_id"] = space_encoded
        results["space_encoder"] = le_space
        results["n_space"] = len(le_space.classes_)

    print(f"Final dataset shape: {results['X'].shape}\n")
    return results
# ============================================================================
# FASTERKAN IMPLEMENTATION
# ============================================================================

class RSWAF(nn.Module):
    """Reflectional Switch Activation Function"""
    def __init__(self, in_features, num_grids=32, grid_range=(-2, 2), denominator=0.15):
        super().__init__()
        low, high = grid_range
        centers = torch.linspace(low, high, num_grids)
        centers = centers.unsqueeze(0).repeat(in_features, 1)
        self.centers = nn.Parameter(centers)
        self.h = nn.Parameter(torch.full((in_features, 1), float(denominator)))
        self.weights = nn.Parameter(torch.randn(in_features, num_grids) * 0.01)
        self.num_grids = num_grids

    def forward(self, x):
        B, D = x.shape
        x_exp = x.unsqueeze(2).expand(B, D, self.num_grids)
        centers = self.centers.unsqueeze(0).expand(B, D, self.num_grids)
        h = self.h.unsqueeze(0).expand(B, D, 1)

        normalized = (x_exp - centers) / (h + 1e-9)
        activation = 1.0 - torch.tanh(normalized) ** 2

        weights = self.weights.unsqueeze(0).expand(B, D, self.num_grids)
        output = (activation * weights).sum(dim=2)
        return output

class FasterKANLayer(nn.Module):
    """Single FasterKAN layer"""
    def __init__(self, in_features, out_features, num_grids=32,
                 grid_range=(-2, 2), denominator=0.15):
        super().__init__()
        self.linear_in = nn.Linear(in_features, in_features)
        self.rswaf = RSWAF(in_features, num_grids, grid_range, denominator)
        self.linear_out = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.rswaf(x)
        x = self.linear_out(x)
        return x

class FasterKAN(nn.Module):
    """FasterKAN network with optional dropout"""
    def __init__(self, layers_list, num_grids=32, grid_range=(-2, 2),
                 denominator=0.15, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_list) - 1):
            self.layers.append(
                FasterKANLayer(layers_list[i], layers_list[i+1],
                             num_grids, grid_range, denominator)
            )
            if dropout > 0 and i < len(layers_list) - 2:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_regression(model, train_loader, val_loader, device,
                    epochs=150, lr=1e-3, weight_decay=1e-3):
    """Train regression model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y_reg"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y_reg"].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, LR: {new_lr:.6f}")

        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

def train_classification(model, train_loader, val_loader, device,
                        epochs=100, lr=1e-3, weight_decay=1e-3):
    """Train classification model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            x = batch["x"].to(device)

            if "y_floor_bldg" in batch:
                y = batch["y_floor_bldg"].to(device)
            else:
                y = batch["y_space"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(pred, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                if "y_floor_bldg" in batch:
                    y = batch["y_floor_bldg"].to(device)
                else:
                    y = batch["y_space"].to(device)

                pred = model(x)
                _, predicted = torch.max(pred, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_acc = val_correct / val_total

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}, LR: {new_lr:.6f}")

        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    return model, history

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_classification(model, loader, device):
    """Evaluate classification performance"""
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            if "y_floor_bldg" in batch:
                y = batch["y_floor_bldg"]
            else:
                y = batch["y_space"]

            pred = model(x).cpu()
            _, predicted = torch.max(pred, 1)
            all_preds.append(predicted)
            all_trues.append(y)

    preds = torch.cat(all_preds).numpy()
    trues = torch.cat(all_trues).numpy()

    accuracy = accuracy_score(trues, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': preds,
        'true_labels': trues
    }
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def prepare_dataset():
    """Merge training and validation data into single dataset"""
    os.makedirs("data/UJIIndoor", exist_ok=True)

    train_path = "data/UJIIndoor/trainingData.csv"
    val_path = "data/UJIIndoor/validationData.csv"
    output_path = "data/UJIIndoor/UJIIndoorDataset.csv"

    if not os.path.exists(train_path):
        print(f"❌ {train_path} not found!")
        return False

    print("Merging training and validation datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    merged_df = pd.concat([train_df, val_df], ignore_index=True)
    merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged dataset saved: {len(merged_df)} total samples\n")
    return True
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print()

    if not prepare_dataset():
        return

    print("Loading UJIIndoorLoc dataset...")
    data = load_and_preprocess_ujiindoor(
        "data/UJIIndoor/UJIIndoorDataset.csv",
        task="all"
    )

    input_dim = data["n_features"]
    print(f"✅ Input dimension: {input_dim} (correctly filtered!)")
    print(f"Number of samples: {data['X'].shape[0]}")
    print(f"Number of floor&building classes: {data['n_floor_bldg']}")
    print(f"Number of space ID classes: {data['n_space']}\n")

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data["X"])

    # ========================================================================
    # TASK 1: COORDINATE REGRESSION
    # ========================================================================
    print("="*70)
    print("TASK 1: COORDINATE REGRESSION")
    print("="*70)

    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(data["coords"])

    print(f"\nOriginal coord range: {data['coords'].min(axis=0)} to {data['coords'].max(axis=0)}")
    print(f"Scaled coord range: {coords_scaled.min(axis=0)} to {coords_scaled.max(axis=0)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, coords_scaled, test_size=0.3, random_state=42
    )

    train_dataset = WiFiDataset(X_train, y_reg=y_train)
    test_dataset = WiFiDataset(X_test, y_reg=y_test)

    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Training Optimized FasterKAN...")
    model_reg = FasterKAN([input_dim, 400, 400, 2], num_grids=32,
                          grid_range=[-2, 2]).to(device)

    model_reg, history_reg = train_regression(
        model_reg, train_loader, test_loader, device,
        epochs=200, lr=1e-3, weight_decay=1e-3
    )

    # Evaluate
    model_reg.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y_reg"]
            pred = model_reg(x).cpu()
            all_preds.append(pred)
            all_trues.append(y)

    preds_scaled = torch.cat(all_preds).numpy()
    trues_scaled = torch.cat(all_trues).numpy()

    preds_original = coord_scaler.inverse_transform(preds_scaled) + data["coords_min"]
    trues_original = coord_scaler.inverse_transform(trues_scaled) + data["coords_min"]

    errors = np.sqrt(((preds_original - trues_original) ** 2).sum(axis=1))
    mean_error = errors.mean()

    print(f"\n{'='*70}")
    print(f"✅ OPTIMIZED FasterKAN - Mean Position Error: {mean_error:.2f} m")
    print(f"   Median Error: {np.median(errors):.2f} m")
    print(f"   75th Percentile: {np.percentile(errors, 75):.2f} m")
    print(f"   90th Percentile: {np.percentile(errors, 90):.2f} m")
    print(f"{'='*70}\n")

    # ========================================================================
    # TASK 2: FLOOR & BUILDING CLASSIFICATION
    # ========================================================================
    print("="*70)
    print("TASK 2: FLOOR & BUILDING CLASSIFICATION")
    print("="*70)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, data["floor_bldg"]))

    X_train_fb = X_scaled[train_idx]
    X_test_fb = X_scaled[test_idx]
    y_train_fb = data["floor_bldg"][train_idx]
    y_test_fb = data["floor_bldg"][test_idx]

    train_dataset_fb = WiFiDataset(X_train_fb, y_floor_bldg=y_train_fb)
    test_dataset_fb = WiFiDataset(X_test_fb, y_floor_bldg=y_test_fb)

    train_loader_fb = DataLoader(train_dataset_fb, batch_size=batch_size, shuffle=True)
    test_loader_fb = DataLoader(test_dataset_fb, batch_size=batch_size, shuffle=False)

    print("\nTraining FasterKAN for Floor & Building...")
    model_fb = FasterKAN(
        [input_dim, 400, 400, data["n_floor_bldg"]], num_grids=32
    ).to(device)
    model_fb, _ = train_classification(
        model_fb, train_loader_fb, test_loader_fb, device, epochs=100
    )

    results_fb = evaluate_classification(model_fb, test_loader_fb, device)
    print(f"\n{'='*70}")
    print(f"Floor & Building Classification Results:")
    print(f"  Accuracy: {results_fb['accuracy']*100:.2f}%")
    print(f"  Precision: {results_fb['precision']:.4f}")
    print(f"  Recall: {results_fb['recall']:.4f}")
    print(f"  F1-Score: {results_fb['f1']:.4f}")
    print(f"{'='*70}\n")

    # ========================================================================
    # TASK 3: SPACE ID CLASSIFICATION
    # ========================================================================
    print("="*70)
    print("TASK 3: SPACE ID CLASSIFICATION")
    print("="*70)

    sss_space = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx_sp, test_idx_sp = next(sss_space.split(X_scaled, data["space_id"]))

    X_train_sp = X_scaled[train_idx_sp]
    X_test_sp = X_scaled[test_idx_sp]
    y_train_sp = data["space_id"][train_idx_sp]
    y_test_sp = data["space_id"][test_idx_sp]

    train_dataset_sp = WiFiDataset(X_train_sp, y_space=y_train_sp)
    test_dataset_sp = WiFiDataset(X_test_sp, y_space=y_test_sp)

    train_loader_sp = DataLoader(train_dataset_sp, batch_size=batch_size, shuffle=True)
    test_loader_sp = DataLoader(test_dataset_sp, batch_size=batch_size, shuffle=False)

    print("\nTraining FasterKAN for Space ID...")
    model_sp = FasterKAN(
        [input_dim, 400, 400, data["n_space"]], num_grids=32
    ).to(device)
    model_sp, _ = train_classification(
        model_sp, train_loader_sp, test_loader_sp, device, epochs=100
    )

    results_sp = evaluate_classification(model_sp, test_loader_sp, device)
    print(f"\n{'='*70}")
    print(f"Space ID Classification Results:")
    print(f"  Accuracy: {results_sp['accuracy']*100:.2f}%")
    print(f"  Precision: {results_sp['precision']:.4f}")
    print(f"  Recall: {results_sp['recall']:.4f}")
    print(f"  F1-Score: {results_sp['f1']:.4f}")
    print(f"{'='*70}\n")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nCoordinate Regression:")
    print(f"  Mean Position Error: {mean_error:.2f} m")
    print(f"\nFloor & Building Classification: {results_fb['accuracy']*100:.2f}%")
    print(f"Space ID Classification: {results_sp['accuracy']*100:.2f}%")

    print("\n" + "="*70)
    print("COMPARISON WITH PAPER")
    print("="*70)
    improvement_fb = results_fb['accuracy']*100 - 99.0
    improvement_sp = results_sp['accuracy']*100 - 71.0
    improvement_pos = 3.56 - mean_error

    print(f"Floor & Building: 99.00% (Paper) vs {results_fb['accuracy']*100:.2f}% (Ours) ", end="")
    print(f"[{improvement_fb:+.2f}%]")
    print(f"Space ID:         71.00% (Paper) vs {results_sp['accuracy']*100:.2f}% (Ours) ", end="")
    print(f"[{improvement_sp:+.2f}%]")
    print(f"Position Error:   3.56m  (Paper) vs {mean_error:.2f}m (Ours) ", end="")
    print(f"[{improvement_pos:+.2f}m]")
    print("="*70)

if __name__ == "__main__":
    main()