import torch
from torchvision.models import resnet50, ResNet50_Weights

class KCClassifier(torch.nn.Module):
    def __init__(self, loss_func):
        super().__init__()

        # ResNet backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze backbone parameters

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        child_count = 0
        for child in self.resnet.children():
            child_count += 1
            if child_count < 7:
                for param in child.parameters():
                    param.requires_grad = False

        # Adding layers
        fc_inputs = self.resnet.fc.in_features

        if loss_func == "focal_loss":
            last_layer = torch.nn.Softmax(dim=1) 
        elif loss_func == "nllloss":
            # last_layer = torch.nn.LogSoftmax(dim=1)
            last_layer = torch.nn.Sigmoid()
        elif loss_func == "bceloss":
            last_layer = torch.nn.Sigmoid()
        elif loss_func == "celoss":
            last_layer = None

        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_inputs, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            # torch.nn.Linear(256, 2),
            torch.nn.Linear(256, 1),
            last_layer
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        return x
    

if __name__ == "__main__":
    kcclassifier = KCClassifier(loss_func="bceloss").cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    out = kcclassifier(input_tensor)
    print(out)