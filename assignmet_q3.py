import torch
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import zipfile

# from google.colab import drive
# drive.mount('/content/drive')
# cudnn.benchmark = True
plt.ion()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Extract the zip file
zip_file_path = 'hymenoptera_data.zip'
extract_dir = 'extracted_data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

data_dir = os.path.join(extract_dir, 'hymenoptera_data')
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def extract_resnet_features(model, dataloader):
    features = []
    model.eval()
    with torch.no_grad():
        for inputs, levels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())

    return torch.from_numpy(np.concatenate(features, axis=0))


# Load the pre-trained ResNet18 model
resnet18 = models.resnet18(pretrained=True)
# Remove the final fully connected layer
resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-1]))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets['train'].transform = data_transforms

features = extract_resnet_features(resnet18, dataloaders['train'])

X_train = features.numpy()  # Convert tensor to numpy array

# Extract labels from the dataloader
y_train = np.array(image_datasets['train'].targets)
X_tained = X_train.reshape(X_train.shape[0], -1)
features_test = extract_resnet_features(resnet18, dataloaders['val'])

# Converting tensor to numpy array
X_test = features_test.numpy()
X_test = X_test.reshape(X_test.shape[0], -1)

# Extracting labels from the dataloader
y_test = np.array(image_datasets['val'].targets)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

features = features.squeeze()
print(features.shape)

svm_param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1]}

rf_param_grid = {'max_depth': [10, 20, 30, None],
                 'n_estimators': [50, 100, 200]}

svm_model = SVC(kernel='rbf')
svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=5, n_jobs=-1)
svm_grid_search.fit(X_tained, y_train)

svm_pred = svm_grid_search.best_estimator_.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)
print("SVM F1 Score:", svm_f1)
rf_model = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

rf_pred = rf_grid_search.best_estimator_.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest F1 Score:", rf_f1)