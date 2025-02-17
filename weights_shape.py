import torch

pth_path = r"C:\myyyy\Siddhant\chicken.pth.tar"  # Path to your checkpoint
checkpoint = torch.load(pth_path, map_location="cpu")  # Load checkpoint

# Iterate over models in checkpoint
for model_name in ["generator", "discriminator", "kp_detector"]:
    print(f"\n=== {model_name.upper()} Weights ===")
    state_dict = checkpoint[model_name]  # Extract weights dictionary
    for layer, weights in state_dict.items():
        print(f"{layer}: {weights.shape}")  # Print shape
