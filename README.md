# RWTH GDMC 2026

Minecraft settlement generation with FQVAE-based building reconstruction.

This project is part of the RWTH Aachen University practical lab **Applications of AI in Games**. The project investigates whether a **Factorised Quantised Variational AutoEncoder (FQVAE)**, originally designed for Minecraft terrain generation, can be adapted to voxel-based building reconstruction and integrated into a GDMC settlement-generation pipeline.

The project is intended as a submission to the **GDMC (Generative Design in Minecraft) Competition 2026**.

## Project Overview

The current pipeline:

1. Terrain analysis
2. Building generation with a trained FQVAE
3. Building placement
4. Route graph planning
5. Settlement generation

The project is built for **Minecraft Java Edition 1.21.11**.

## Current Status

Terrain analysis, placement, and route planning are functional. The FQVAE preserves the coarse geometry and overall silhouette of many input houses, but the reconstructed buildings are not yet consistently complete enough for a finished competition agent.

## Requirements

- Python `>=3.10,<3.14.1`
- Poetry
- Minecraft Java Edition **1.21.11**
- GDMC HTTP interface/mod compatible with GDPC
- A trained FQVAE checkpoint

## Installation

Install Poetry if necessary:

```powershell
pip install poetry
```

Install the dependencies:

```powershell
poetry install
```

Run the project through Poetry:

```powershell
poetry run python main.py
```

## Model and Data Setup

The default `HouseGenerator` configuration uses:

```python
MODEL_NAME = "fqgan"
MODEL_DIR = "logdir_64_64_3DCraft_full_grouped4"
CHECKPOINT_STEP = 2500
INPUT_CHUNK_SIZE = 24
```

Make sure that:

- the checkpoint is available at the location expected by `get_model`;
- `BLOCK_INDEX_TO_NAME_DICT_PATH` points to the correct mapping;
- `group_dict4.json` is present in `DATA_DIR`; and
- the grouping dictionary matches the checkpoint's training configuration.

Input schematics are currently loaded from:

```text
minecraft_gen/templates/
```


## Running the Settlement Generator

1. Start Minecraft Java Edition 1.21.11.
2. Open a world with the GDMC HTTP interface enabled.
3. Set a build area, for example:

```text
/setbuildarea ~0 ~0 ~0 ~200 ~200 ~200
```

4. Run:

```powershell
poetry run python main.py
```

Keep Minecraft and the GDMC HTTP interface running while the script executes.


## Repository Structure

```text
CODE_SUBMISSION/
├── config/
│   └── paths.py
│
├── data/
│   ├── gdcm2025/
│   ├── biome_index_to_name_config1.json
│   ├── block_index_to_name_config2.json
│   ├── block_index_to_name_config3.json
│   ├── create_legacy_112_to_121_dict.py
│   ├── group_data.py
│   ├── group_dict_config1.json
│   ├── group_dict_config2.json
│   ├── group_dict_config3.json
│   └── process_raw_data.py
│
├── minecraft_gen/
│   ├── __init__.py
│   ├── gen_buildings.py
│   ├── HouseBuilder.py
│   ├── HouseGenerator.py
│   ├── PathBuilder.py
│   └── WorldMap.py
│
├── model/
│   ├── __init__.py
│   ├── fq_models.py
│   ├── style_structure_codebook_labeling.py
│   └── train.py
│
├── notebooks/
│   ├── analyse_data.ipynb
│   └── analyse_stats.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── log_utils.py
│   ├── model_utils.py
│   └── visualization_utils.py
│
├── main.py
├── pyproject.toml
├── README.md
```

The exact model and processed-data locations depend on `config/paths.py`.

## Development Notes

Add a dependency:

```powershell
poetry add <package>
```

Add a development dependency:

```powershell
poetry add --group dev <package>
```

Update dependencies:

```powershell
poetry update
```

Keep `poetry.lock` committed so that all contributors use the same dependency versions.

## Authors

- Hoang Dung Dang

## Acknowledgements

The terrain-adaptive settlement pipeline is based on ideas from [**Gaussianly Filtered**](https://github.com/IsaacBraamGit/gdmc2025), the winning project of the GDMC 2025 competition. The building model is based on the [FQVAE implementation](https://github.com/TimMerino1710/DisentanglingMinecraft) from the paper *Procedural Content Generation in Minecraft via Disentangled Representation Learning Models*. 
