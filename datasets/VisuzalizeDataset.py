import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from google.colab import drive

# -------------------------------------------------------------------------
# Step 1: Mount Drive & Ensure Archives Exist
# -------------------------------------------------------------------------
print("[+] Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

DRIVE_DIR = Path('/content/drive/MyDrive/sen12ms_cr')
LOCAL_DIR = Path('/content/data/sen12ms_cr')
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

s2_clear_archive = DRIVE_DIR / "ROIs1158_spring_s2.tar.gz"
s1_archive = DRIVE_DIR / "ROIs1158_spring_s1.tar.gz"
s2_cloudy_archive = DRIVE_DIR / "ROIs1158_spring_s2_cloudy.tar.gz"

CLEAR_ARCHIVE_URL = "ftp://m1554803:m1554803@dataserv.ub.tum.de/ROIs1158_spring_s2.tar.gz"

# Check if the clear archive is missing or corrupted (< 100MB check)
if not s2_clear_archive.exists() or s2_clear_archive.stat().st_size < 100 * 1024 * 1024:
    print(f"[!] {s2_clear_archive.name} missing or corrupted. Re-downloading (~20 GB)...")
    if s2_clear_archive.exists():
        s2_clear_archive.unlink() # Delete partial file
    !wget -O "{s2_clear_archive}" "{CLEAR_ARCHIVE_URL}"

# -------------------------------------------------------------------------
# Step 2: Unpack Archives Safely
# -------------------------------------------------------------------------
print("[+] Extracting dataset archives into local memory...")

# Extract with overwrite check
!tar -xzf "{s1_archive}" -C "{LOCAL_DIR}"
!tar -xzf "{s2_cloudy_archive}" -C "{LOCAL_DIR}"
!tar -xzf "{s2_clear_archive}" -C "{LOCAL_DIR}"

print("[✓] Unpacking complete!")

# -------------------------------------------------------------------------
# Step 3: Match and Visualize First 4 Triplets
# -------------------------------------------------------------------------
def visualize_dataset_samples(root_dir="/content/data/sen12ms_cr", num_samples=4):
    root = Path(root_dir)
    
    # Locate all S1 image patches
    s1_files = sorted(list(root.rglob("*_s1_*.tif")))
    
    if not s1_files:
        print("[!] No S1 TIFF files found in local storage!")
        return

    matched_triplets = []
    
    for s1_p in s1_files:
        s1_str = str(s1_p)
        
        # Exact folder & string mapping for SEN12MS-CR structure
        s2_cloudy_str = s1_str.replace("ROIs1158_spring_s1", "ROIs1158_spring_s2_cloudy").replace("_s1_", "_s2_cloudy_")
        s2_clear_str = s1_str.replace("ROIs1158_spring_s1", "ROIs1158_spring_s2").replace("_s1_", "_s2_")
        
        if os.path.exists(s2_cloudy_str) and os.path.exists(s2_clear_str):
            matched_triplets.append((s1_str, s2_cloudy_str, s2_clear_str))
            if len(matched_triplets) == num_samples:
                break

    if not matched_triplets:
        print("[!] Matching failed. Extracted folders present:")
        for sub in root.glob("*"):
            if sub.is_dir():
                print(f"  └── {sub.name}")
        return

    print(f"[+] Found {len(matched_triplets)} valid triplets. Rendering grid...")

    fig, axes = plt.subplots(len(matched_triplets), 3, figsize=(12, 3.5 * len(matched_triplets)))
    
    for idx, (s1_p, s2_cloudy_p, s2_clear_p) in enumerate(matched_triplets):
        # Load SAR Radar (VV Band)
        with rasterio.open(s1_p) as src:
            s1_img = np.clip(src.read(1), -25, 0)

        # Load S2 Cloudy RGB (B4=Red, B3=Green, B2=Blue)
        with rasterio.open(s2_cloudy_p) as src:
            rgb_cloudy = np.clip(src.read([4, 3, 2]).transpose(1, 2, 0) / 3000.0, 0, 1)

        # Load S2 Ground Truth Clear RGB
        with rasterio.open(s2_clear_p) as src:
            rgb_clear = np.clip(src.read([4, 3, 2]).transpose(1, 2, 0) / 3000.0, 0, 1)

        ax_sar = axes[idx, 0] if len(matched_triplets) > 1 else axes[0]
        ax_cloudy = axes[idx, 1] if len(matched_triplets) > 1 else axes[1]
        ax_clear = axes[idx, 2] if len(matched_triplets) > 1 else axes[2]

        ax_sar.imshow(s1_img, cmap='gray')
        ax_sar.set_title(f"Sample {idx+1}: Radar SAR (S1)")
        ax_sar.axis('off')

        ax_cloudy.imshow(rgb_cloudy)
        ax_cloudy.set_title(f"Sample {idx+1}: Cloudy Optical (S2)")
        ax_cloudy.axis('off')

        ax_clear.imshow(rgb_clear)
        ax_clear.set_title(f"Sample {idx+1}: Ground Truth Clear (S2)")
        ax_clear.axis('off')

    plt.tight_layout()
    plt.show()

# Run visualization
visualize_dataset_samples(root_dir="/content/data/sen12ms_cr", num_samples=4)