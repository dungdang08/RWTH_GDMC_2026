from pathlib import Path

# RWTH_GDMC_2026/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main folders
DATA_DIR = PROJECT_ROOT / "data"

# Raw data directories
GDMC_2025_DATA_DIR = DATA_DIR / "gdcm2025"
# GDMC_2025_PROCESSED_DATA_DIR = GDMC_2025_DATA_DIR / "processed" # (this dir is deleted -> use DATA_BUILDING_GDMC2025_DIR instead)
GDMC_2025_RAW_DATA_DIR = GDMC_2025_DATA_DIR / "raw"
HOUSE_3D_CRAFT_DATA_DIR = DATA_DIR / "3DCraft_houses" / "houses"
HOUSE_3D_CRAFT_DATA_DIR_FILTERED = DATA_DIR / "3DCraft_houses" / "filtered_houses"
HOUSE_3D_CRAFT_DATA_DIR_GABLE_ROOF = DATA_DIR / "3DCraft_houses" / "gable_roof"
# Dataset files
DISENTANGLING_DATA_DIR = DATA_DIR / "fqvae_original"
# ORIGINAL_MAPPINGS_PATH = DISENTANGLING_DATA_DIR / "24_newdataset_mappings3.pt"

DATA_PROCESSED_FOR_CONVERTER_DIR = DATA_DIR / "processed_npz" # this is where the intermediate .npz files are stored for loading into the BlockBiomeConverter, which then saves the final .pt files in DATA_PROCESSED_DIR
DATA_PROCESSED_DIR = DATA_DIR / "processed" # this is where the final processed .pt dataset files and mappings are stored for training

# DATASET_NAME is defined in data/process_raw_data.py when creating the dataset
# DATASET_NAME = "24_gdmc_2025_combined"
# DATASET_NAME = "24_3DCraft_grouped2_full"
# DATASET_NAME = "24_3DCraft_grouped3_full"
DATASET_NAME = "24_3DCraft_grouped4_full"
NEW_DATASET_FOR_CONVERTER_PATH = DATA_PROCESSED_FOR_CONVERTER_DIR / f"{DATASET_NAME}.npz" 
NEW_DATASET_PATH = DATA_PROCESSED_DIR / f"{DATASET_NAME}.pt"
NEW_MAPPINGS_PATH = DATA_PROCESSED_DIR / f"{DATASET_NAME}_mappings.pt"

# Paths for generating buildings
DATA_BUILDINGS_DIR = PROJECT_ROOT / "buildings"
DATA_BUILDINGS_GDMC2025_DIR = DATA_BUILDINGS_DIR / "gdmc2025"

# Relative paths for data processing
BASE_LOG_DIR = "model_logs"

BLOCK_INDEX_TO_NAME_DICT_PATH = DATA_DIR / "block_index_to_name.json"
META_BLOCK_INDEX_TO_NAME_DICT_PATH = DATA_DIR / "meta_block_index_to_name.json"
BIOME_INDEX_TO_NAME_DICT_PATH = DATA_DIR / "biome_index_to_name.json"