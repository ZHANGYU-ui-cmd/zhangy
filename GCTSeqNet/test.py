import torch
import torch.nn.functional as F

a = [0.5, 0.9, 0.8, 0.1, 0.2, -1]
ious = torch.nonzero(torch.tensor(a) == -1.0).reshape(-1)
print(len(ious))

x = torch.zeros(5532, 512)
x[5][0] = 0.000001
if torch.all(x[5].eq(0)):
    print(1)