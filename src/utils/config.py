import torch

state_dict = torch.load("saved_models/best_model.pth", map_location="cpu")

print(type(state_dict))
print(list(state_dict.keys())[:10])