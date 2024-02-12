from example import vit_base_patch16_224
import torch
import torchvision
from  torch import nn
from  torch.utils.data import DataLoader



dataset=torchvision.datasets.CIFAR10(
    root="../train1",train=True,

    transform=torchvision.transforms.Compose([
   torchvision.transforms.Resize((224,224)),
   torchvision.transforms.ToTensor()]
    ),
download=True
)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataload=DataLoader(dataset,64)
model=vit_base_patch16_224()
opt=torch.optim.Adam(model.parameters(),0.01)
loss=nn.CrossEntropyLoss()

# 1.dataload  2. opt  3. nn.crossentropyloss 4. model(a,b,c,d)




for epoch in range(32):
    for i,(x,target) in enumerate(dataload):

        opt.zero_grad()
        x=x.to(device)
        target=target.to(device)
        model=model.to(device)

        y=model(x)
        lost=loss(y,target)

        lost.backward()
        opt.step()
        if (i % 10 == 0):
            row = torch.argmax(y, dim=1)
            print(row)
            print(target)
            common_elements = torch.sum(row == target)
            print("第", epoch, "次训练", "正确率为", common_elements / 64)
            print(lost)
#1. zerograd 2.x,target,model to (device) 3.output and lost 4.lost.backward,opt.step



















