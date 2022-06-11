import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_For_Age(nn.Module):
    def __init__(self, model_name='resnet50', pretrained = True):
        super().__init__()
        self.resnet = timm.create_model(model_name = model_name, pretrained = pretrained) 
        self.num_features = self.resnet.num_features
        
        self.age_fc = nn.Sequential(
                nn.Linear(in_features = self.num_features, out_features = 256),
                nn.ReLU(),
                nn.Linear(in_features = 256, out_features = 1)
            )

        self.gender_fc = nn.Sequential(
                nn.Linear(in_features = self.num_features, out_features = 1),
                nn.Sigmoid()
            )


    def forward_head(self, x):
        x = self.resnet.global_pool(x)
        age = self.age_fc(x)
        gender = self.gender_fc(x)
        return age, gender

    def forward(self, x):
        x = self.resnet.forward_features(x)
        age, gender = self.forward_head(x)
        return age, gender



class Vit_For_Age(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224', pretrained = True, drop_path_rate = 0.05):
        super(Vit_For_Age, self).__init__()

        self.vit = timm.create_model(model_name = model_name, pretrained = pretrained, drop_path_rate = drop_path_rate)  # tranformer model
        self.embed_dim = self.vit.embed_dim

        self.age_fc = nn.Sequential(
                nn.Linear(in_features = self.embed_dim, out_features = 96),
                nn.ReLU(),
                nn.Linear(in_features = 96, out_features = 1) 
            ) # regression

        self.gender_fc = nn.Sequential(
                nn.Linear(in_features = self.embed_dim, out_features = 1),
                nn.Sigmoid()
            ) # binary classification

    def forward(self,x):
         x = self.vit.forward_features(x)
         age = self.age_fc(x)
         gender = self.gender_fc(x)
         return age, gender





'''
support model list:
'vit_small_patch16_224'
'vit_base_patch16_224'
'vit_tiny_patch16_224'
'resnet50'
'resnet34'
'resnet18'
'',

'''
def build_model(model_name='resnet50', model_type = 'resnet' ,pretrained=True):
    if model_type == 'resnet':
        model = ResNet_For_Age(model_name=model_name, pretrained=pretrained)
    elif model_type == 'transformer':
        model = Vit_For_Age(model_name=model_name, pretrained=pretrained)

    return model
