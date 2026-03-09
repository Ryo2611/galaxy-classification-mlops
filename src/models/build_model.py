import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import wandb
import numpy as np

class GalaxyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Define target columns based on Kaggle dataset (Class 1-11 probabilities, total 37)
        self.target_cols = [col for col in self.labels_df.columns if col != 'GalaxyID']
        
        # Filter dataframe for images that actually exist 
        # Kaggle files are .jpg
        available_images = [f.replace('.jpg', '') for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
        # Convert IDs to string to ensure safe matching
        self.labels_df['GalaxyID'] = self.labels_df['GalaxyID'].astype(str)
        
        print(f"Total labels in CSV: {len(self.labels_df)}")
        print(f"Total JPG images found in directory: {len(available_images)}")
        
        # Filter
        self.labels_df = self.labels_df[self.labels_df['GalaxyID'].isin(available_images)].reset_index(drop=True)
        print(f"Dataset mapped size: {len(self.labels_df)} images.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        galaxy_id = str(self.labels_df.iloc[idx]['GalaxyID'])
        img_name = os.path.join(self.img_dir, f"{galaxy_id}.jpg")
        
        # Load image (ensuring RGB)
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Handle rare case where file got deleted after init
            print(f"Warning: Image not found {img_name}")
            # Return dummy but 0 loss target or crash gracefully. We'll let it crash for now to surface errors.
            raise FileNotFoundError(f"Missing image: {img_name}")

        if self.transform:
            image = self.transform(image)

        # Get all 37 probability labels as a float tensor
        targets = self.labels_df.iloc[idx][self.target_cols].values.astype(np.float32)
        
        return image, torch.tensor(targets)

def train_baseline(csv_path, img_dir, epochs=15, batch_size=32, patience=3):
    # Initialize Weights & Biases
    wandb.init(
        project="galaxy-classification-portfolio",
        name="resnet50-baseline",
        config={
            "architecture": "ResNet-50",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss_function": "MSE"
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard ResNet image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Simple Data Augmentation
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No paired images and labels found! Did you run downloading/preprocessing?")
        return
        
    print(f"Dataset loaded with {len(dataset)} samples. Predicting {len(dataset.target_cols)} probabilities.")
    
    # 80/20 train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize architecture
    model = models.resnet50(pretrained=False) # Or weights='DEFAULT'
    num_ftrs = model.fc.in_features
    # We use Sigmoid for probability regression [0, 1] across all 37 classes
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, len(dataset.target_cols)),
        nn.Sigmoid() 
    )
    model = model.to(device)

    # Loss and Optimizer (MSE is common for the Kaggle Galaxy Zoo probability metric)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "../models/baseline_resnet50_best.pth"
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                 inputs, targets = inputs.to(device), targets.to(device)
                 outputs = model(inputs)
                 loss = criterion(outputs, targets)
                 val_loss += loss.item() * inputs.size(0)
                 
        epoch_val_loss = val_loss / val_size if val_size > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": epoch_val_loss
        })
        
        # Checkpoint: Save best model
        os.makedirs("models", exist_ok=True)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = "models/baseline_resnet50_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    print(f"Training complete. Best model saved to {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to training labels CSV")
    parser.add_argument("--img_dir", type=str, default="data/processed/rgb_images", help="Directory of processed PNGs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    args = parser.parse_args()
    
    train_baseline(args.csv_path, args.img_dir, args.epochs, args.batch_size, args.patience)
