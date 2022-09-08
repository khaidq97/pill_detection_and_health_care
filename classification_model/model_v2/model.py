import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from model_v2.diagnose_encoder import DiagnoseEncoder
from model_v2.image_encoder import ImageEncoder

class DrugClassificationModel_v2(nn.Module):
    def __init__(self, backbone, num_classes, bert_model, use_diagnose=True, use_drugname=True, use_additional=True):
        super(DrugClassificationModel_v2, self).__init__()
        self.num_classes = num_classes
        self.use_diagnose = use_diagnose
        self.use_drugname = use_drugname
        self.use_additional = use_additional
        
        self.input_dim = 256
        if use_diagnose:
            self.input_dim += 256
        if self.use_drugname:
            self.input_dim += 256
        if self.use_additional:
            self.input_dim += 256

        self.image_encoder = ImageEncoder(backbone)
        self.diagnose_encoder = DiagnoseEncoder(bert_model)
        self.drugname_encoder = nn.Linear(143, 256)
        self.additional_encoder = nn.Linear(83+5, 256) # 83 #61
        
        
        # self.linear = nn.Linear(self.input_dim, 256)
        # self.drop_out = nn.Dropout(0.2)
        self.out = nn.Linear(self.input_dim, num_classes)
    
    def forward(self, image_input, diagnose_encoder, drugnames_input, bbox_input, doctor_input, quantity_input):
        encode_features = []
        # image
        image_feature = self.image_encoder(image_input)
        encode_features.append(image_feature)
        # diagnose
        if self.use_diagnose:
            diagnose_feature = self.diagnose_encoder(diagnose_encoder)
            encode_features.append(diagnose_feature)
        # drugname
        if self.use_drugname:
            drugname_feature = F.relu(self.drugname_encoder(drugnames_input))
            encode_features.append(drugname_feature)
        # additional
        if self.use_additional:
            # additional_feature = torch.concat((bbox_input, doctor_input, quantity_input), dim=1)
            additional_feature = torch.concat((doctor_input, quantity_input), dim=1)
            additional_feature = F.relu(self.additional_encoder(additional_feature))
            encode_features.append(additional_feature)

        x = torch.cat(encode_features, dim=1)
        # x = F.relu(self.linear(x))
        # x = self.drop_out(x)
        out = self.out(x)

        return F.log_softmax(out, dim=1)
        # return out