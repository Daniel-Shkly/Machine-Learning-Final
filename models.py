from typing import Tuple
from lightning.pytorch import LightningModule
from torch import nn
import torch
from torch import Tensor
import torchmetrics

class BaseModel(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes)
        self.model = self.build_model()
        
    def build_model(self):
        raise Exception("Not yet implemented")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.model(x)
    
    def loss(self, logits, target):
        return nn.functional.cross_entropy(logits, target)
    
    def shared_step(self, mode:str, batch:Tuple[Tensor, Tensor], batch_index:int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.accuracy(output, target)
        self.log(f"{mode}_step_acc", self.accuracy, prog_bar=True)
        self.log(f"{mode}_step_loss", loss, prog_bar=False)
        return loss
    
    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)
    
    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)
    
    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)

              
class DeepConvNet(BaseModel):        
    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(131072, 5)
        )
    

class DeeperConvNet(BaseModel):        
    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(65536, 5)
        )


class DeepestConvNet(BaseModel):        
    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32768, 5)
        )

        
