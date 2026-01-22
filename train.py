import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    lr=0.001,
    model_name='vehicle'          
):
    """
    Trains the given model and saves it with a name-specific filename.
    
    Args:
        model: PyTorch model (ResNet18 or Custom CNN)
        train_loader, val_loader: DataLoaders
        num_epochs: Number of epochs to train
        lr: Learning rate
        model_name: String identifier (e.g. 'resnet', 'custom_cnn')
                    Used in print messages and filename
    
    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    print(f"Starting training: {model_name} ({num_epochs} epochs)")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    save_path = f"{model_name}_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to: {save_path}\n")

    return model