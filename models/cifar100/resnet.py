import torch
import torch.nn as nn

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()

        self.lr = lr
        self.device = device
        self.num_classes = num_classes
        self.image_channels = 3
        self.in_channels = 16

        self.conv1 = nn.Conv2d(self.image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        self.gn1 = nn.GroupNorm(2, 16)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            #first block
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(),
            
            #second block
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(),

            #third block
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.GroupNorm(2, 16),
            nn.ReLU()
            
        )
        
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
        )    
        

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
        )

        self.layer2_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(32),
            nn.GroupNorm(2, 32),
            nn.ReLU(),

        )

        self.identity_2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=1,stride=2,bias=False),
                #nn.BatchNorm2d(32),
                nn.GroupNorm(2, 32),
            )
            

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
        
        )  

        self.layer3_2 = nn.Sequential(
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
             
        )

        self.layer3_3 = nn.Sequential(
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
             
        )

        self.identity_3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=1,stride=2,bias=False),
                #nn.BatchNorm2d(64),
                nn.GroupNorm(2, 64),
            )



        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.num_classes)

        self.size = self.model_size()

    def forward(self, x):
        #base
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.gn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        
        #16
        x = self.layer1(x)

        #32
        identity = x.clone()
        x = self.layer2_1(x)
        identity = self.identity_2(identity)
        x = x.clone() + identity
        x = self.relu(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        
        #64
        identity = x.clone()
        x = self.layer3(x)
        identity = self.identity_3(identity)
        x = x.clone() + identity
        x = self.relu(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
    

