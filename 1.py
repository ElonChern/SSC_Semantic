from torch_scatter import scatter_max
import torch
src = torch.Tensor([0,1,2,7])
index = torch.tensor([0,1,0,2])
out = src.new_zeros((10))


y_true = torch.Tensor([0,1,2,7])
print(y_true)
nonempty_idx = torch.Tensor([0,1,1,0])
y_true[nonempty_idx == 0] =0
print(y_true)




out,_ = scatter_max(src,index,out=out)
print(out)

a = torch.Tensor([[1,2],[3,4]])
b = torch.Tensor([[0,1],[1,0]])
a = a.view(-1)
a = a.reshape(2,2)
# a[b==0]=0
print("a = {}".format(a))


bb = torch.randn(3,2,1)
print(bb.shape)
cc = bb.T
print(cc.shape)


data = torch.tensor([[1,2],[3,4]])
def count_label(max,min,data):
    assert max>min
    ONE = torch.ones_like(data)
    ZERO = torch.zeros_like(data)
    CONDITION = (data<max) * (data>min)
    CAL_LABEL = torch.where(CONDITION, ONE, ZERO)
    num = torch.sum(CAL_LABEL)  
    return num  

num = count_label(3,1,data)
print (num)

data = torch.tensor([[1,2],[3,4]])
print(data[1,:])
print(data[0,:])
import numpy as np
x = np.array([1,2,3])
y = np.array([4])
z = np.insert(x,3,y)
print(z)