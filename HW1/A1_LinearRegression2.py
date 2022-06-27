import torch

##############################
## Specify the matrix X
## Dimensions: X (3x3)
##############################
X = torch.Tensor([[0,0,1],[1,1,1],[4,2,1]])
y = torch.Tensor([[0],[1],[1]])
print(X)
print(y)

# Solution
##############################
## Use one of the ways to compute the result
##############################
res1 = torch.solve(y, X)
print(res1[0])

# l = torch.matmul(torch.t(X), X)
# r = torch.matmul(torch.t(X), y)
# print(torch.inverse(l))
# res1 = torch.matmul(torch.inverse(l),r)
# print(res1)