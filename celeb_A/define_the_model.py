import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
from collections import OrderedDict
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the model
class ResNet_50(nn.Module):
    def __init__(self, output_classes=1):
        super(ResNet_50, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, output_classes)
        )
    
    def forward(self, x, apply_sigmoid=True):
        logits = self.resnet_model(x)
        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
        
def ResNet_18(output_classes=3):
  """
    Load a pretrained ResNet-18 model from PyTorch with a specific last dense layer with a dedicated number of neurons.
    Previous layers are already frozen to update and train only the last classification layer.
    
    :param output_classes: The number of output classes in our classification problem, defining the number of neurons in the last layer.
    
    :type output_classes: integer
  """
  # Load a pretrained ResNet-18 model thanks to PyTorch
  resnet_model = models.resnet18(pretrained = True)
  # Change the last dense layer to fit our problematic needs
  resnet_model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(512, 128)), ('relu', nn.ReLU()),
    ('fc2', nn.Linear(128, output_classes)), ('output', nn.Sigmoid())
  ]))
  
  return resnet_model
# Utility Functions
def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def fit_model(model, X_data, y_data, EPOCHS=5, BATCH_SIZE=32):
    optimizer = torch.optim.Adam(model.parameters())
    #error = nn.MSELoss()
    #error = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss for binary classification
    error = nn.BCELoss()
    model.train()
    n = X_data.shape[0]

    for epoch in range(EPOCHS):
        obsIDs = np.arange(X_data.shape[0])
        np.random.shuffle(obsIDs)
        
        for batch_start in range(0, n, BATCH_SIZE):
            if batch_start + BATCH_SIZE > n:
                break

            Curr_obsIDs = obsIDs[batch_start:batch_start + BATCH_SIZE]
            var_X_batch = X_data[Curr_obsIDs,:,:,:].float().to(DEVICE)
            var_y_batch = y_data[Curr_obsIDs,:].float().to(DEVICE)

            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            print(f'Epoch {epoch}, Batch {batch_start}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "./modelFinal.pytorchModel")

def showRGBImage(LodID, X, Y):
    LocImage = (X[LodID,:,:,:] * 255).astype(int)
    LocTitle = 'Y=' + str(int(Y[LodID,0]))
    plt.figure()
    plt.imshow(LocImage)
    plt.title(LocTitle)
    plt.savefig('Observation_' + str(LodID) + '.png')
    plt.show()