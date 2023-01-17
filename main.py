import CNN




exec("CNN.py")


correct = 0
total = 0
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
