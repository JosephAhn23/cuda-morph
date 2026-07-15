import torch
from PHASE3_COMPLETE import morphos_backend_phase3

model = torch.nn.Linear(10, 5)
compiled = torch.compile(model, backend=morphos_backend_phase3)

x = torch.randn(10)
output = compiled(x)
print(output)
