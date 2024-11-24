import torch
save_path = 'training_output/20241120_131155/models/epoch_100.pth'
checkpoint = torch.load(save_path)
print(checkpoint['model_state_dict']['teamworkencoder'])
# print(checkpoint['model_state_dict']['adhocencoder'])
# print(checkpoint['model_state_dict']['returnnet'])
# print(checkpoint['model_state_dict']['goaldecoder'])
