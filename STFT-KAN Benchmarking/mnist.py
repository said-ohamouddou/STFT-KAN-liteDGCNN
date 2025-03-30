import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import torchvision.models as models

# Assuming all custom layers are imported:
from kans import KANLayer, KALNLayer, JacobiKANLayer, GRAMLayer, FastKANLayer, WavKANLayer,\
    BernsteinKANLayer, NaiveFourierKANLayer
    
from stft_kan import STFTKANLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MNIST dataset with specific transform for feature extraction
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match typical CNN input
    transforms.Grayscale(3),  # Convert to 3 channels (replicate grayscale)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Feature extraction batches can be larger since we're not training the feature extractor
feature_batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=feature_batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=feature_batch_size, shuffle=False)

# Load pre-trained ResNet18 for feature extraction
feature_extractor = models.resnet18(pretrained=True)
# Remove the final fully connected layer to get features instead of class scores
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()  # Set to evaluation mode

# Extract features from all images
def extract_all_features():
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    print("Extracting features from training data...")
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            # Get features and flatten them
            features = feature_extractor(images)
            # Proper reshaping: [batch_size, channels, 1, 1] -> [batch_size, channels]
            features = features.reshape(features.size(0), -1)
            train_features.append(features.cpu())
            train_labels.append(labels)
    
    print("Extracting features from test data...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # Get features and flatten them
            features = feature_extractor(images)
            # Proper reshaping: [batch_size, channels, 1, 1] -> [batch_size, channels]
            features = features.reshape(features.size(0), -1)
            test_features.append(features.cpu())
            test_labels.append(labels)
    
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)
    
    print(f"Feature extraction complete. Feature shape: {train_features.shape}")
    return train_features, train_labels, test_features, test_labels

# Extract and save features
print("Starting feature extraction...")
extraction_start = time.perf_counter()
train_features, train_labels, test_features, test_labels = extract_all_features()
extraction_end = time.perf_counter()
print(f"Feature extraction time: {extraction_end - extraction_start:.2f} seconds")

# Create feature datasets
train_feature_dataset = TensorDataset(train_features, train_labels)
test_feature_dataset = TensorDataset(test_features, test_labels)

# Hyperparameters for classifier training
batch_size = 128
learning_rate = 0.001
num_epochs = 40
input_size = train_features.shape[1]  # Feature dimension from ResNet (512)
hidden_size = 64
num_classes = 10
print(input_size)
# Print feature dimensions to debug
print(f"Feature input size: {input_size}")

# DataLoaders for classifier training
train_loader = DataLoader(train_feature_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_feature_dataset, batch_size=batch_size, shuffle=False)

# Layer configurations with only the specific parameters (no input/output dimensions)
import random

# Set a random seed for reproducibility
random.seed(4)
# Create a list of valid combinations based on the given conditions

layer_configs = {
    "MLP": (nn.Linear, {}),
    "KANLayer": (KANLayer, {"grid_size": 1, "spline_order": 0}),
    "KALNLayer": (KALNLayer, {"degree": 0}),
    "GRAMLayer": (GRAMLayer, {"degree": 0}),
    "FastKANLayer": (FastKANLayer, {"grid_min": 0, "grid_max": 1, "num_grids": 2}),
    "WavKANLayer": (WavKANLayer, {"wavelet_type": "mexican_hat"}),
    "ChebyKAN": (JacobiKANLayer, {"degree": 0, "a": 1/2, "b": 1/2}),
    "BernsteinKANLayer": (BernsteinKANLayer, {"degree": 0}),
    "NaiveFourierKANLayer": (NaiveFourierKANLayer, {"gridsize": 1}),
    "STFTKANLayer(3,40,5)": (STFTKANLayer, {"gridsize": 3, "window_size": 40, "stride": 5}),
    "STFTKANLayer(2,40,10)": (STFTKANLayer, {"gridsize": 2, "window_size": 40, "stride": 10}),
    "STFTKANLayer(6,50,9)": (STFTKANLayer, {"gridsize": 6, "window_size": 50, "stride": 9}),
    "STFTKANLayer(5,50,10)": (STFTKANLayer, {"gridsize": 5, "window_size": 50, "stride": 10}),
    "STFTKANLayer(7,60,10)": (STFTKANLayer, {"gridsize": 7, "window_size": 60, "stride": 10}),
    "STFTKANLayer(20,60,10)": (STFTKANLayer, {"gridsize": 20, "window_size": 60, "stride": 10}),
    "STFTKANLayer(12,60,5)": (STFTKANLayer, {"gridsize": 12, "window_size": 60, "stride": 5}),
    "STFTKANLayer(40,160,20)": (STFTKANLayer, {"gridsize": 40, "window_size": 160, "stride": 20}),
    "STFTKANLayer(10,20,20)": (STFTKANLayer, {"gridsize": 10, "window_size": 20, "stride": 20}),
    "STFTKANLayer(5,20,20)": (STFTKANLayer, {"gridsize": 5, "window_size": 20, "stride": 20}),
    "STFTKANLayer(5,20,25)": (STFTKANLayer, {"gridsize": 5, "window_size": 20, "stride": 25}),
    "STFTKANLayer(5,20,22)": (STFTKANLayer, {"gridsize": 5, "window_size": 20, "stride": 22})
 }



class Net(nn.Module):
    def __init__(self, base_layer_class, base_layer_params, hidden_size, num_classes=10, base_type="MLP"):
        super(Net, self).__init__()
        
        self.base_type = base_type
        
        # Add required input/output parameters based on layer type
        input_param_name = self._get_input_param_name(base_layer_class)
        output_param_name = self._get_output_param_name(base_layer_class)
        
        # First layer (input to hidden)
        fc1_params = base_layer_params.copy()
        if input_param_name:
            fc1_params[input_param_name] = input_size
        if output_param_name:
            fc1_params[output_param_name] = hidden_size
        self.fc1 = base_layer_class(**fc1_params)
        
        # Second layer (hidden to output)
        fc2_params = base_layer_params.copy()
        if input_param_name:
            fc2_params[input_param_name] = hidden_size
        if output_param_name:
            fc2_params[output_param_name] = num_classes
        self.fc2 = base_layer_class(**fc2_params)
        
        self.relu = nn.ReLU()

    def _get_input_param_name(self, layer_class):
        """Determine the appropriate input parameter name for the layer class"""
        if  layer_class in [nn.Linear, WavKANLayer]:
            return "in_features"
        elif layer_class in [KANLayer]:
            return "input_features"
        elif layer_class in [FastKANLayer, JacobiKANLayer, BernsteinKANLayer]:
            return "input_dim"
        elif layer_class in [GRAMLayer]:
            return "in_channels"
        elif layer_class == KALNLayer:
            return "input_features"
        elif layer_class in [NaiveFourierKANLayer, STFTKANLayer]:
            return "inputdim"
        return None

    def _get_output_param_name(self, layer_class):
        """Determine the appropriate output parameter name for the layer class"""
        if layer_class in [nn.Linear, WavKANLayer]:
            return "out_features"
        elif layer_class in [KANLayer]:
            return "output_features"
        elif layer_class in [FastKANLayer, JacobiKANLayer, BernsteinKANLayer]:
            return "output_dim"
        elif layer_class in [GRAMLayer]:
            return "out_channels"
        elif layer_class == KALNLayer:
            return "output_features"
        elif layer_class in [NaiveFourierKANLayer, STFTKANLayer]:
            return "outdim"

        return None

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten for fully connected layers
        out = self.fc1(x)
        if self.base_type == "MLP":
            out = self.relu(out)
        out = self.fc2(out)
        return out

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

results = []
best_test_acc = 0
best_model_info = None

for layer_name, (layer_class, layer_params) in layer_configs.items():
    model = Net(base_layer_class=layer_class, 
                base_layer_params=layer_params, 
                hidden_size=hidden_size, 
                num_classes=num_classes,
                base_type=layer_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f'Training {layer_name}...')
    
    # Track epoch-wise accuracies
    epoch_results = []
    
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Evaluate after each epoch
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        
        # Store epoch results
        epoch_results.append({
            "Epoch": epoch + 1,
            "Train_Accuracy": train_acc,
            "Test_Accuracy": test_acc
        })
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Find the epoch with the best test accuracy for this model
    best_epoch_for_model = max(epoch_results, key=lambda x: x["Test_Accuracy"])
    best_train_acc = best_epoch_for_model["Train_Accuracy"]
    best_test_acc_for_model = best_epoch_for_model["Test_Accuracy"]
    best_epoch_num = best_epoch_for_model["Epoch"]
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Store all results for reference
    results.append({
        "LayerType": layer_name,
        "Best_Epoch": best_epoch_num,
        "Best_Train_Accuracy": best_train_acc,
        "Best_Test_Accuracy": best_test_acc_for_model,
        "Num_Params": num_params,
        "Train_Time_sec": execution_time
    })
    
    # Update best model information if this model has better test accuracy
    if best_test_acc_for_model > best_test_acc:
        best_test_acc = best_test_acc_for_model
        best_model_info = {
            "LayerType": layer_name,
            "Best_Epoch": best_epoch_num,
            "Best_Train_Accuracy": best_train_acc,
            "Best_Test_Accuracy": best_test_acc_for_model,
            "Num_Params": num_params,
            "Train_Time_sec": execution_time
        }
    
    print(f'{layer_name} - Best Epoch: {best_epoch_num}, Train Acc: {best_train_acc:.4f}, Test Acc: {best_test_acc_for_model:.4f}, Params: {num_params}, Time: {execution_time:.2f}s')
    print('-' * 80)

df_results = pd.DataFrame(results)
print("\nFinal Results:")
print(df_results)

# Save results to CSV
df_results.to_csv('kan_mnist_comparison_results.csv', index=False)
print("Results saved to kan_mnist_comparison_results.csv")
