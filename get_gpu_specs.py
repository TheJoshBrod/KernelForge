import torch
props = torch.cuda.get_device_properties(0)
print(props)
