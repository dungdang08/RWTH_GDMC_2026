import json
import pandas as pd

from config.paths import META_BLOCK_INDEX_TO_NAME_DICT_PATH, DATA_DIR

BLOCK_GROUPS = (
    "AIR",
    "GLASS",
    "DOOR",
    "STAIR",

    "FLOOR",
    "STRUCTURE",
    # "SOLID_WALL",
    # "ROOF",
    "INNER_DECORATION",
    "OUTER_DECORATION",
    
    "IGNORE_SOLID",
)

FLOOR_KEYWORDS = {
    "dirt", 
    "grass_block", 
    "snow_block",
    "sand", 
    "gravel",
    "ice",
    "bedrock",

    "deepslate",
    "slab",
    "andesite",
    "diorite",
    "granite",


}

STRUCTURE_KEYWORDS = {
    "planks",
    "wood",
    "log",
    "stem",
    
    "sandstone",
    "cobblestone",
    "bricks",
    "concrete",
    "clay",

    "prismarine",
    "terracotta",
    "quartz",
    "netherrack" ,
    "purpur",
    "polished",

    "bone_block",
    "coal_block",
    "diamond_block",
    "emerald_block",
    "gold_block",
    "iron_block",
    "lapis_block",
    "magma_block",
    "obsidian",
    "redstone_block",
    "slime_block",
}

INNER_DECORATION_KEYWORDS = {
    # lights
    "torch",
    "lantern",
    "lamp",
    "glowstone",
    "candle",
    "end_rod",
    "sea_lantern",

    # furniture / interior
    "carpet",
    "banner",
    "sign",
    "painting",
    "item_frame",
    "pot",
    "bookshelf",
    "chest",
    "barrel",
    "crafting_table",
    "furnace",
    "anvil",
    "cauldron",
    "bell",
    "bed",
    "box",
    "wool",
    "brewing_stand",
    "cake",
    "enchanting_table",
    "jukebox",
    "note_block",
    "lever",

    # small details
    "button",
    "pressure_plate",
    "plate",
    "rail",
    "chain",
    "ladder",
    "head",
    "skull",
}

OUTER_DECORATION_KEYWORDS = {
    # exterior structure details
    "fence",
    "gate",
    "wall",
    "trapdoor",
}

def export_group_blocks():
    """
    Export the block grouping as a json file.
    """
    block_index_to_name = {}
    if META_BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
        with open(META_BLOCK_INDEX_TO_NAME_DICT_PATH, "r") as f:
            block_index_to_name = json.load(f)
    
    
    block_name_to_group = {}
    group_dict = {group: [] for group in BLOCK_GROUPS}
    for _, block_name in block_index_to_name.items():
        block_name = block_name.lower()
        block_name = block_name.replace("minecraft:", "")
        tmp_name_list = block_name.split("_")

        if "air" in tmp_name_list:
            block_name_to_group[block_name] = "AIR"
        elif "glass" in tmp_name_list:
            block_name_to_group[block_name] = "GLASS"
        elif "door" in tmp_name_list:
            block_name_to_group[block_name] = "DOOR"
        elif "stair" in tmp_name_list or "stairs" in tmp_name_list:
            block_name_to_group[block_name] = "STAIR"
        elif any(all(k in tmp_name_list for k in keyword.split("_")) for keyword in FLOOR_KEYWORDS):
            block_name_to_group[block_name] = "FLOOR"
        elif any(all(k in tmp_name_list for k in keyword.split("_")) for keyword in STRUCTURE_KEYWORDS):
            block_name_to_group[block_name] = "STRUCTURE"
        elif any(all(k in tmp_name_list for k in keyword.split("_")) for keyword in INNER_DECORATION_KEYWORDS):
            block_name_to_group[block_name] = "INNER_DECORATION"
        elif any(all(k in tmp_name_list for k in keyword.split("_")) for keyword in OUTER_DECORATION_KEYWORDS):
            block_name_to_group[block_name] = "OUTER_DECORATION"
        else:
            block_name_to_group[block_name] = "IGNORE_SOLID"
        group_dict[block_name_to_group[block_name]].append(block_name)        
        
    # Export the grouping as a json file
    # with open(DATA_DIR / "block_name_to_group.json", "w") as f:
    #     json.dump(block_name_to_group, f, indent=4)
    with open(DATA_DIR / "group_dict.json", "w") as f:
        json.dump(group_dict, f, indent=4)



# -------------------------------------------------------------------------
# SECOND GROUPING METHOD: MANUAL GROUPING BASED ON BLOCK TYPES
# -------------------------------------------------------------------------
BLOCK_GROUPS2 = {
    "structure_block": ["bone_block", "quartz_block", "diamond_block", "emerald_block", "gold_block", "iron_block", "lapis_block", "magma_block", "nether_wart_block", "purpur_block", "redstone_block", "slime_block", "prismarine", "netherrack"],
    "ground_block": ["coal_block", "grass_block", "snow_block", "dirt", "coarse_dirt", "podzol", "gravel", "ice", "sand", "mycelium", "bedrock", "clay"],
    "rock_block": ["andesite", "diorite", "granite"],
    "light_block": ["glowstone", "lantern", "lamp", "shroomlight", "end_rod"],
    "flower_block": ["allium", "azure_bluet", "orchid", "dandelion", "tulip", "flower", "sunflower", "daisy", "peony", "poppy", "lilac", "rose_bush"],
    "plant_block": ["grass", "fern", "dead_bush", "vines", "sugar_cane", "cactus", "mushroom", "chorus_plant", "chorus_flower", "melon_stem", "pumpkin_stem", "nether_wart", "sapling"],
    "furniture_block": ["bed", "bookshelf", "crafting_table", "enchanting_table", "jukebox", "note_block", "brewing_stand", "flower_pot", "carpet", "obsidian", "wool", "box"],
    
    "glass_pane": ["glass_pane"],
    "pressure_plate": ["pressure_plate"],
    "trapdoor": ["trapdoor"],

    "door": ["door"],
    "fence": ["fence"],
    "glass": ["glass"],
    "planks": ["planks"],
    "log": ["log"],
    "slab": ["slab"],
    "terracotta": ["terracotta"],
    "concrete": ["concrete"],
    "stairs": ["stairs"],
    "pillar": ["pillar"],
    "bricks": ["bricks"],
    "wall": ["wall"],
    "ore": ["ore"],
    "stone": ["stone", "sandstone", "cobblestone"],

    "air": []
}

def export_group_blocks2():
    """
    Export the block grouping as a json file.
    """
    block_index_to_name = {}
    if META_BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
        with open(META_BLOCK_INDEX_TO_NAME_DICT_PATH, "r") as f:
            block_index_to_name = json.load(f)
    
    block_name_to_group = {}
    group_dict = {group: [] for group in BLOCK_GROUPS2}
    for _, block_name in block_index_to_name.items():
        block_name = block_name.lower()
        block_name = block_name.replace("minecraft", "")
        for group, keywords in BLOCK_GROUPS2.items():
            if any(f":{k}" in block_name or f"_{k}" in block_name for k in keywords):
                block_name_to_group[block_name] = group
                group_dict[group].append(block_name.replace(":", ""))
                break
        # If no keywords match, assign to "air" group
        if block_name not in block_name_to_group:
            block_name_to_group[block_name] = "air"
            group_dict["air"].append(block_name.replace(":", ""))

        
    # Export the grouping as a json file
    # with open(DATA_DIR / "block_name_to_group.json", "w") as f:
    #     json.dump(block_name_to_group, f, indent=4)
    with open(DATA_DIR / "group_dict2.json", "w") as f:
        json.dump(group_dict, f, indent=4)

def export_group_blocks3():
    """
    Export the block grouping as a json file.
    """
    # Read the csv file
    df = pd.read_csv(DATA_DIR / "group_dict3.csv")
    group_dict = {}
    print(df['Categories'].unique())
    for group in df['Categories'].unique():
        item_list = df[df['Categories'] == group]["Items"].iloc[0].split(",")
        item_list = [item.strip() for item in item_list]
        group_dict[str(group)] = item_list
    with open(DATA_DIR / "group_dict3.json", "w") as f:
        json.dump(group_dict, f, indent=4)
    

if __name__ == "__main__":
    # export_group_blocks()
    # export_group_blocks2()
    export_group_blocks3()