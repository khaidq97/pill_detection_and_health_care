from torchvision.models import resnet50,  efficientnet_v2_l, EfficientNet_V2_L_Weights
import torchvision
import torch.nn as nn
import torch
from transformers import AutoFeatureExtractor, SwinModel
import torch.nn.functional as F

class ProjectionModule(nn.Module):
    def __init__(self, input_channels):
        super(ProjectionModule, self).__init__()
        self.linear_1 = nn.Linear(input_channels, 320)
        self.linear_2 = nn.Linear(320, 512)
        self.linear_3 = nn.Linear(512, 768)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))

        return x
    
class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(ImageEncoder, self).__init__()
        self.backbone = backbone

        if self.backbone=="resnet50":
            self.resnet50 = resnet50(pretrained=True) # 512
            self.resnet50.fc = nn.Sequential(nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024, 256),
                                        nn.ReLU()
                                        )
        elif self.backbone=="efficientnet":
            self.efficientnet_l = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT, progress=True).features # 1280
            self.eff_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            self.eff_drop = nn.Dropout(0.2)
            self.eff_linear = nn.Linear(1280, 256)            
                
        elif self.backbone=="swin":     
            self.project_module = ProjectionModule(300)  
            self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.swin_linear = nn.Sequential(nn.Linear(768,256),
                                        nn.ReLU(), 
                                        # nn.BatchNorm1d(256),
                                        # nn.Dropout(0.2)
                                        )

    def forward(self, image_input):
        if self.backbone=="resnet50":
            feature = self.resnet50(image_input)
            return feature
        elif self.backbone=="efficientnet":
            feature = self.efficientnet_l(image_input)
            feature = self.eff_avgpool(feature)
            feature = self.eff_drop(feature)
            feature = self.eff_linear(feature.squeeze())

            return feature
        elif self.backbone=="swin":
            x = self.swin(image_input)
            feature = self.swin_linear(x[1])
            
            hidden_s = x.last_hidden_state
            hidden_s = hidden_s.transpose(1,2)
            batch_size, num_channels, sequence_length = hidden_s.shape
            hidden_s = hidden_s.reshape(batch_size, 256, int((num_channels/256)*sequence_length))
            pm_feature = self.project_module(hidden_s)
            
            return feature, pm_feature

