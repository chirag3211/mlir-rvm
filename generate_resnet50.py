import torch
import torchvision.models as models
from torch_mlir import fx

def generate_mlir():
    print("Loading PyTorch ResNet-50 (v2 weights)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()

    # Use a deterministic solid gray image instead of random noise
    example_input = torch.ones(1, 3, 224, 224) * 0.5

    # 1. Get the baseline PyTorch prediction
    with torch.no_grad():
        pt_out = model(example_input)
        example_input.numpy().astype("float32").tofile("input.bin")
        pt_out.numpy().astype("float32").tofile("ref_logits.bin")
        winning_class = torch.argmax(pt_out).item()
        print(f"-> PYTORCH EXPECTED ARGMAX CLASS: {winning_class}")

    # 2. Export to MLIR
    print("Exporting model to MLIR Linalg on Tensors...")
    with torch.no_grad():
        module = fx.export_and_import(
            model, example_input,
            output_type="linalg-on-tensors"
        )

    with open("resnet50_v2.mlir", "w") as f:
        f.write(str(module))
    print("Success! Saved MLIR to resnet50_v2.mlir")

if __name__ == "__main__":
    generate_mlir()
