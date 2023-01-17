import train
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch


lr = 0.0001
epochs = 20
batchsize = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
testset = dsets.MNIST(root='dataset/',
                      train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                      download=True)

testloader = DataLoader(testset,batch_size= batchsize,shuffle=True)



correct = 0
total = 0
model = torch.load("CNN.pt") ##모델 틀 불러오기 대용
model.load_state_dict(torch.load("CNN_dict.pt"))  ## 모델 가중치 불러오기
model.eval()

with torch.no_grad():
    for image,label in testloader:
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _,output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index == y).sum().float()

    print("Accuracy",(100.0*correct)/total)
