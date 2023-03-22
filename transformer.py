from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks.mlp import MLPBlock
import config


DEVICE = 'cuda'

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):
    def __init__(self,
            embed_size,
            heads):
        super(AttentionBlock, self).__init__();

        self.__embed_size = embed_size;
        self.__heads = heads;
        self.__head_dim = embed_size // heads;

        self.__values = nn.Linear(self.__head_dim, self.__head_dim, bias=False);
        self.__keys = nn.Linear(self.__head_dim, self.__head_dim, bias=False);
        self.__queries = nn.Linear(self.__head_dim, self.__head_dim, bias=False);
        self.__fc_out = nn.Linear(self.__heads * self.__head_dim, embed_size);

    def forward(self, values, keys, queries):
        N = queries.shape[0];

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1];

        values = values.reshape(N, value_len, self.__heads, self.__head_dim);
        keys = keys.reshape(N, key_len, self.__heads, self.__head_dim);
        queries = queries.reshape(N, query_len, self.__heads, self.__head_dim);

        values = self.__values(values);
        keys = self.__keys(keys);
        queries = self.__queries(queries);


        e = torch.einsum('nqhd,nkhd->nhqk', [queries, keys]);
        
        attention = torch.softmax(e/((self.__embed_size)**(1/2)), dim=3);

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.__heads*self.__head_dim);
        out = self.__fc_out(out);

        return out;

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, ) -> None:
        super().__init__();
        self.__block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 'same'),
            nn.LayerNorm(),
            nn.ReLU()
        )
    def forward(self, x):
        return self.__block(x);
    
class Patch(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embedding_size, dropout = 0.0) -> None:
        super().__init__();

        self.__patch_size = patch_size;
        num_patches = (img_size // self.__patch_size)**2;
        self.__patch_embedding = nn.Conv2d(in_channels, embedding_size,self.__patch_size, stride = self.__patch_size);
        self.__pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embedding_size));
        self.__cls_embedding = nn.Parameter(torch.zeros(1, 1, embedding_size));

        self.__dropout = nn.Dropout(dropout);
    
    def forward(self, x):
        x = self.__patch_embedding(x);
        x = x.flatten(2).permute(0,2,1);
        embedding = torch.cat([self.__cls_embedding.expand(config.BATCH_SIZE*2, -1,-1), x], dim=1);
        embedding = embedding + self.__pos_embedding.expand(config.BATCH_SIZE*2, -1, -1);
        embedding = self.__dropout(embedding);
        return embedding;



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        super(TransformerBlock, self).__init__();

        self.__attention = SABlock(embed_size, heads, dropout);
        self.__norm1 = nn.LayerNorm(embed_size);
        self.__norm2 = nn.LayerNorm(embed_size);

        self.__ff = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion*embed_size, embed_size),
            nn.Dropout(dropout),
        )

        #self.mlp = MLPBlock(embed_size, forward_expansion*embed_size, dropout)

        self.__dropout = nn.Dropout(dropout);

    
    def forward(self, x):
        x = x + self.__attention(self.__norm1(x));
        x = x + self.__ff(self.__norm2(x));
        return x;

class Encoder(nn.Module):
    def __init__(self, 
    embed_size,
    heads,
    forward_expansion,
    dropout,) -> None:
        super().__init__();
        self.layers = nn.Sequential(*[TransformerBlock(embed_size,heads,dropout, forward_expansion) for _ in range(12)])


    def forward(self, x):
        
        return self.layers(x);

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device) -> None:
        super().__init__()
        self.attention = SelfAttention(embed_size, heads);
        self.norm = nn.LayerNorm(embed_size);

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        );

        self.dropout = nn.Dropout(dropout);
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask);
        query = self.dropout(self.norm(attention + x));
        out = self.transformer_block(value, key, query, src_mask);
        return out;


class Decoder(nn.Module):
    def __init__(self,
    trg_vocab_size,
    embed_size,
    num_layers,
    heads,
    forward_expansion,
    dropout,
    device,
    max_length):
        super().__init__();

        self.embedding = nn.Embedding(trg_vocab_size, embed_size);
        self.pos_embedding = nn.Embedding(max_length, embed_size);

        self.final = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, trg_vocab_size)
        )

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size);
        self.dropout = nn.Dropout(dropout);
        self.device = device;

    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape;
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device);
        x = self.dropout((self.pos_embedding(positions) + self.embedding(x)));
        for l in self.layers:
             x = l(x, enc_out, enc_out, src_mask, trg_mask);
            
        out = self.fc_out(x);


class ViT(nn.Module):
    def __init__(
        self,
        in_channels, 
        img_size, 
        patch_size, 
        embedding_size,
        heads,
        dropout,
        forward_expansion,
        num_classes
        ) -> None:
        super().__init__();
        num_patches = (img_size//patch_size)**2;
        self.patch_size = patch_size;

        self.input_layer = nn.Linear(in_channels*(patch_size**2), embedding_size)
        self.__patch_embedding = Patch(in_channels, img_size, patch_size, embedding_size, dropout);
        self.__transformer_encoder = Encoder(embedding_size, heads, forward_expansion, dropout);
        self.__classifier = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_size))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embedding_size))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.__patch_embedding(x);
        B, T, _ = x.shape
        #x = self.input_layer(x)

        # Add CLS token and positional encoding
        # cls_token = self.cls_token.repeat(B, 1, 1)
        # x = torch.cat([cls_token, x], dim=1)
        # x = x + self.pos_embedding[:,:T+1];

        x = self.dropout(x)

        enc = self.__transformer_encoder(x);
        return enc;


def test():
    v = ViT(3,32,16,1024,8,0.5,4,10);
    summary(ViT(3,224,16,768,8,0.5,4,1000), (3,224,224), device='cpu');
    t = torch.randn((8,1,256,256));
    out = v(t);
    print(out.shape);

def train_step(e, model, critic, optim, loader):
    pbar = tqdm(loader);
    print(('\n' + '%s'*2)%('Epoch', 'Loss'));
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    mean_loss = [];
    for (img, lbl) in pbar:
        img,lbl = img.to(DEVICE),lbl.to(DEVICE);
        optimizer.zero_grad(set_to_none = True)

        # forward + backward + optimize
        outputs = model(img)
        loss = criterion(outputs, lbl)
        loss.backward()
        optimizer.step()

        # print statistics
        mean_loss.append(loss.item());
        pbar.set_description(('%10s' + '%10.4g')%(e, np.mean(mean_loss)));

def test_step(e, model, loader, writer):
    with torch.no_grad():
        pbar = tqdm(loader);
        print(('\n' + '%s'*5)%('Epoch', 'Loss','Prec', 'Rec', 'F1'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        mean_loss = [];
        mean_f1 = [];
        mean_prec = [];
        mean_rec = [];
        for (img, lbl) in pbar:
            img,lbl = img.to(DEVICE),lbl.to(DEVICE);

            # forward + backward + optimize
            outputs = model(img)
            loss = criterion(outputs, lbl)
            pred = torch.argmax(outputs,dim=1).detach().cpu().numpy();
            prec,rec,f1,_ = precision_recall_fscore_support(lbl.cpu().detach(), pred, average='macro')

            # print statistics
            mean_loss.append(loss.item());
            mean_prec.append(prec);
            mean_rec.append(rec);
            mean_f1.append(f1);
            pbar.set_description(('%10s' + '%10.4g'*4)%(e, np.mean(mean_loss), np.mean(mean_prec), np.mean(mean_rec), np.mean(mean_f1)));
            writer.add_scalar('Loss', np.mean(mean_loss), e);
            writer.add_scalar('F1', np.mean(mean_f1), e);

def get_lr(e,baselr):
    return (min(e,40)/40)*baselr;

if __name__ == "__main__":
    #test();
    v = ViT(3,32,4,768,8,0.5,4,10).to(DEVICE);
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    ss = Subset(trainset,np.arange(0,64*5));
    trainloader = torch.utils.data.DataLoader(ss, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
                                        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(v.parameters(), lr=5e-4)
    writer = SummaryWriter('exp');
    for e in range(1,1000):
        train_step(e, v, criterion, optimizer, trainloader);
        test_step(e,v,trainloader, writer);

