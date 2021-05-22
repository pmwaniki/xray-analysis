import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



def encoder(base_model):
    model,n_out=None,None

    if base_model == "resnet18":
        model = models.resnet18(pretrained=True)
        n_out = 512
    elif base_model == "resnet34":
        model = models.resnet34(pretrained=True)
        n_out = 512
    elif base_model == "resnet50":
        model = models.resnet50(pretrained=True)
        n_out = 2048
    elif base_model == "resnet101":
        model = models.resnet101(pretrained=True)
        n_out = 2048
    elif base_model == "wideresnet50":
        model = models.wide_resnet50_2(pretrained=True)
        n_out = 2048
    elif base_model == "resnet152":
        model = models.resnet152(pretrained=True)
        n_out = 2048

    else:
        NotImplementedError(f"{base_model} not implemented")
    model.fc = nn.Identity()

    return model, n_out


class Net(nn.Module):
    def __init__(self, base_model="resnet18",dropout=0.5,dim_out=5):
        super().__init__()
        self.encoder_model,n_out=encoder(base_model=base_model)
        self.fc=nn.Linear(n_out,dim_out)
        self.dropout=nn.Dropout(p=dropout)
        torch.nn.init.xavier_uniform_(self.fc.weight)
    def forward(self,x):
        x=self.encoder_model(x)
        x=self.dropout(x)
        x=self.fc(x)
        return x

class Net2(nn.Module):
    def __init__(self,base_model,embedding_dim,num_embeddings=18,dropout=0.1,dim_out=5,max_norm=4.0,activation_fun=None,comb_embedd='multiply'):
        super().__init__()
        self.encoder_model, n_out = encoder(base_model=base_model)
        self.embedding=nn.Embedding(num_embeddings,embedding_dim,max_norm=max_norm,norm_type=1)
        self.linear_embedding=nn.Linear(embedding_dim,n_out)
        self.fc = nn.Linear(n_out, dim_out)
        self.dropout = nn.Dropout(p=dropout)
        self.comb_embedd=comb_embedd

        if not self.comb_embedd in ['add','multiply']:
            raise Exception("comb_embedd must be add or multiply")


        torch.nn.init.uniform_(self.embedding.weight,-1.0,1.0)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.linear_embedding.weight)
        self.activation=None
        if activation_fun == 'tanh':
            self.activation=nn.Tanh()
        elif activation_fun == 'sigmoid':
            self.activation=nn.Sigmoid()
        elif activation_fun == 'relu':
            self.activation=nn.ReLU()
        elif activation_fun is None:
            self.activation=nn.Identity()
        else:
            raise NotImplementedError(f'{activation_fun} not implemented.')

        # for param in self.linear_embedding.parameters():
        #     param.requires_grad = False



    def forward(self,x,id):
        x=self.encoder_model(x)
        emb=self.embedding(id)
        emb=self.linear_embedding(emb)
        emb=self.activation(emb)
        if self.comb_embedd == "multiply":
            x=torch.mul(x,emb)
        else:
            x=x+emb
        x=self.dropout(x)
        x=self.fc(x)
        return x


if __name__ == "__main__":
    model=Net(base_model='resnet18').cuda()

    summary(model,(3,224,224))
    model2 = Net2(base_model='resnet18', embedding_dim=8).cuda()
    summary(model2,[(3,224,224),(1,)])



