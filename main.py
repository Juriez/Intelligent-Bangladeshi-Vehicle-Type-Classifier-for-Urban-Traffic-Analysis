from data_collection import download_from_kaggle
from preprocess import get_data_loaders, classes, num_classes
from model import get_model
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    
    train_loader, val_loader, test_loader = get_data_loaders()

    model = get_model(num_classes)

 
    trained_model = train_model(model, train_loader, val_loader)

    
    evaluate_model(trained_model, test_loader, classes)