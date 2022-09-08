import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F

from model_v3.diagnose_encoder import DiagnoseEncoder
from model_v3.image_encoder import ImageEncoder


# Luong attention layer
class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(2)
    
    
class DrugClassificationModel_v3(nn.Module):
    def __init__(self, backbone, num_classes, bert_model, use_diagnose=True, use_drugname=True, use_additional=True):
        super(DrugClassificationModel_v3, self).__init__()
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

        # encode feature
        self.image_encoder = ImageEncoder(backbone)
        self.diagnose_encoder = DiagnoseEncoder(bert_model)
        self.drugname_encoder = nn.Linear(143, 256)
        self.additional_encoder = nn.Linear(2+83+5, 256)
        
        # attention
        self.attn = Attention("concat",768)
        self.attn_linear = nn.Linear(768*2, 256)
        self.drop_out_1 = nn.Dropout(0.2)
        
        self.cls = nn.Linear(self.input_dim+256, num_classes)
    
    def forward(self, image_input, diagnose_encoder, drugnames_input, bbox_input, doctor_input, quantity_input):
        encode_features = []
        # image
        image_feature, image_feature_pm = self.image_encoder(image_input)
        encode_features.append(image_feature)
        # diagnose
        if self.use_diagnose:
            diagnose_feature, sequence_diagnose = self.diagnose_encoder(diagnose_encoder)
            encode_features.append(diagnose_feature)
        # drugname
        if self.use_drugname:
            drugname_feature = F.relu(self.drugname_encoder(drugnames_input))
            encode_features.append(drugname_feature)
        # additional
        if self.use_additional:
            additional_feature = torch.concat((bbox_input, doctor_input, quantity_input), dim=1)
            additional_feature = F.relu(self.additional_encoder(additional_feature))
            encode_features.append(additional_feature)

        pill_feature = torch.cat(encode_features, dim=1)

        # computing attention weights between image feature (Swin) and diagnose feature (PhoBERT)
        attn_weights = self.attn(image_feature_pm, sequence_diagnose)
        context = torch.mul(image_feature_pm, attn_weights)
        
        avg_pool = torch.mean(context, 1)
        max_pool, _ = torch.max(context, 1)        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        attn_features = self.attn_linear(h_conc)

        x = torch.cat((pill_feature, attn_features), dim=1)
        logits = self.cls(x)


        return F.log_softmax(logits, dim=1)
        # return logits