import numpy as np
import torch

model = torch.load('./models/F2T_100.pt')
torch.save(model.state_dict(), './models/f2t_model.pth')