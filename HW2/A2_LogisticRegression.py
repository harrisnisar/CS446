import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
y = torch.Tensor([-1, 1, 1])
w = torch.Tensor([[0.1],[0.1]])
alpha = 1
loss = []

for iter in range(100):
    tmp = torch.exp(torch.matmul(w.t(),X)*(-y))

    ##############################
    ## Use tmp to compute f and g. Instead of summing we average the result, i.e.,
    ## complete only inside torch.mean(...) and don't remove this function
    ## Dimensions: f (scalar); g (2)
    ##############################
    
    f = torch.mean(torch.log(1+tmp))
    g = torch.mean(X*(tmp*-y)/(1+tmp),1)
    print(g)
    print("Loss: %f; ||g||: %f" % (f, torch.norm(g)))
    g = g.view(-1,1)
    w = w - alpha*g
    loss.append(f)

print(w)
plt.plot(loss)
plt.show()

