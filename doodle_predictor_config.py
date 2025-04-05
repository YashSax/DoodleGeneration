from pathlib import Path
import json

class DoodlePredictorConfig:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.config_dict = config
        self.device = config["device"]
        self.stroke_input_size = config["stroke_input_size"]
        self.stroke_embed_size = config["stroke_embed_size"] 
        self.num_attention_heads = config["num_attention_heads"]
        self.num_transformer_blocks = config["num_transformer_blocks"]
        self.transformer_dropout = config["transformer_dropout"]
        self.block_size = config["block_size"]
        self.num_epochs = config["num_epochs"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        
        self.project_name = config["project_name"]
        self.experiment_name = config["experiment_name"]

        self.output_directory = config["output_directory"]
