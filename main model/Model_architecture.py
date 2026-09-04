import torch
import torch.nn as nn
import torchvision

#something is wrong, num_pathches=3136 for image size after resnet is 7x7 and patch size is 4

class CustomEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(CustomEncoder, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Skip connection 1
        skip_1 = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout1(x)
        x = x + skip_1  # Add residual connection
        
        # Skip connection 2
        skip_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = x + skip_2  # Add residual connection
        
        return x
    
class ResNet50_ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=768, num_heads=4, depth=3, num_classes=4):
        super().__init__()

        # **ResNet50 Backbone**
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        self.conv_proj = nn.Conv2d(in_channels=2048, out_channels=embed_dim, kernel_size=patch_size, padding=patch_size // 2)

        num_patches = 8 * 8  # ResNet50 with 224x224 input + conv_proj with patch_size=4 results in 8x8 feature map
        self.position_embed = nn.Embedding(num_patches, embed_dim)
        #self.position_embed = None
        
        # self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.transformer_layers = nn.Sequential(
            *[CustomEncoder(embed_dim, num_heads, 4*embed_dim) for _ in range(depth)]
        )

        self.fc = nn.Linear(embed_dim, num_classes) # num_classes for classification

    def forward(self, x):
        x = self.resnet(x)
        #print(x.shape)
        
        x = self.conv_proj(x) 
        #print(x.shape)

        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        #print(x.shape)

        # if the input image size changes, here'a a given option to adapt the position embedding
        # self.position_embed = None  // should be added before forward
        '''if self.position_embed is None or self.position_embed.size(1) != H*W:
            self.position_embed = nn.Parameter(
                torch.zeros(1, H * W, C, device=x.device)
            )

        x = x + self.position_embed'''

        position_embeddings = self.position_embed(torch.arange(0, H * W).to(x.device))
        x = x + position_embeddings
        #print("After Position Embedding:", x.shape)

        x = self.transformer_layers(x)

        # Change to Global Average Pooling
        x = x.mean(dim=1)
        #print("GAvgPool", x.shape)

        x=self.fc(x)
        #print(x.shape)

        return x

'''class ResNet50WithTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet50_ViT()
        num_ftrs = self.model.vit.embed_dim
        self.fc1 = nn.Linear(num_ftrs, 1) 
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.model(x)  
        x = x[:, 0]  # Extract the class token
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = Binary.apply(x)
        return x  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50_ViT().to(device)
print(model)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)
output = model(x)
print(output.shape)
print(output)'''