from utils.dataset_loader import CXRDataset
from torchvision import transforms

# 🔁 Define transform (same as used in your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 📁 Load dataset
dataset = CXRDataset(root_dir="data/", transform=transform)

# 🔍 List of target filenames you want to find
target_filenames = [
    "CHNCXR_0436_1.png",
    "CHNCXR_0616_1.png",
    "CHNCXR_0596_1.png",
    "CHNCXR_0354_1.png",
    "CHNCXR_0166_0.png",
    "CHNCXR_0090_0.png",
    "CHNCXR_0004_0.png",
    "CHNCXR_0010_0.png",
]

# ✅ Search all targets
found_indices = {}

for target in target_filenames:
    found = False
    for idx, (img_path, _, _) in enumerate(dataset.samples):
        if target in img_path:
            found_indices[target] = idx
            found = True
            break
    if not found:
        found_indices[target] = None

# 📊 Print results
for filename, index in found_indices.items():
    if index is not None:
        print(f"✅ Found '{filename}' at dataset index: {index}")
    else:
        print(f"❌ File '{filename}' not found in dataset.")
