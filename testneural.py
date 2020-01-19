import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ast import literal_eval as make_tuple

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 2 input image channel (we have two input images), 6 output channels, 1x1 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(2, 6, 1)
        #self.conv1 = nn.Conv2d(1,6,1)
        #self.conv1 = nn.Conv2d(2,3,1)
        #self.conv2 = nn.Conv2d(6, 12, 1)
        self.conv1 = nn.Conv1d(2, 3, 1)
        self.conv2 = nn.Conv1d(3, 6, 1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6 * 3, 42)  # 2*3 from image dimension
        self.fc2 = nn.Linear(42, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 1))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    #accuracy = test(model, device, train_loader)
    return 1

def makeXY(theOriginalDataFrame):

    xData = theOriginalDataFrame.drop(columns=[2])
    xList = xData.values.tolist()
    xListLists= []
    for row in xList:
        temp=[]
        for col in row:
            tuple = make_tuple(col)
            list = [tuple[0], tuple[1], tuple[2]]
            temp.append(list)
        xListLists.append(temp)
    #print(xListLists)

    yData = theOriginalDataFrame.drop(columns=[0,1]).values.tolist()

    #print(yData)
    return xListLists, yData

def main():
   
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32
    x = pandas.read_csv('./rgb_two_dominant_with_label.csv', header=None, sep="|")
    #theList = x.values.tolist()
    resX, resY = makeXY(x)

    device = torch.device("cuda" if use_cuda else "cpu")
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    print(net)
    print(torch.FloatTensor(resX))
    params = list(net.parameters())
    #print("The Number Of Learned Parameters: " + str(len(params)))
    #print(params[0].size())
    #input = torch.randn(1, 2, 3, 1)
    input = torch.randn(1,1,2,3)

    target = torch.randn(1)
    target = target.view(1, -1)
    #print("The Target:" + str(target))

    for i in range(20):
        optimizer.zero_grad()
        out = net(torch.FloatTensor(resX))

        criterion = nn.MSELoss()
        loss = criterion(out,torch.FloatTensor(resY))
        loss.backward()
        optimizer.step()
       # if(i%5==0):

        print(loss.item())

    print(out)
    #

    #net.zero_grad()
    #out.backward(torch.randn(1,1))
    '''
    # transform to torch tensors
   
    #Code I am adding
    net = Net()
    params = list(net.parameters())
    criterion = nn.MSELoss()

    # Original code
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    #Adding these arrays!
    training_accuracy_array = []
    testing_accuracy_array = []
    epoch_array = []

    for epoch in range(NumEpochs):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        print(output)
        
        print(loss)
        loss.backward()
        optimizer.step()    # Does the update

        net.zero_grad()
        output = net(input)
        print(output)
        loss = criterion(output, target)
        learning_rate = 0.01
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    
    
    print("\n~~~~~~")
    print(net)
    print("Len Paramaters" + str(len(params)))
    print(params[0].size())
    print("~~~~~~\n")

    #input = torch.rand(1, 1, 3, 1)
    input = torch.FloatTensor([[[[40, 43, 43]]]])
    out = net(input)
    target = torch.FloatTensor([[[[242, 235, 231]]]])  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(out, target)



    print("~~~~~~")
    print("Loss: " + str(loss))
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


    print('\n\nconv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
    print("~~~~~~\n")


    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
    '''

if __name__ == '__main__':
    main()