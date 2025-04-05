import torch
import torch.nn as nn
from doodle_predictor_config import DoodlePredictorConfig

class DoodlePredictor(nn.Module):
    def __init__(self, config: DoodlePredictorConfig):
        super().__init__()
        self.config = config
        self.stroke_encoder_mlp = StrokeEncoderMLP(
            input_size=self.config.stroke_input_size,
            output_size=self.config.stroke_embed_size,
            block_size=self.config.block_size,
            device=self.config.device,
        ).to(self.config.device)

        self.position_embedding_table = nn.Embedding(
            num_embeddings=self.config.block_size,
            embedding_dim=self.config.stroke_embed_size,
        )

        self.stroke_transformer = StrokeTransformer(
            num_transformer_blocks=self.config.num_transformer_blocks,
            num_attention_heads=self.config.num_attention_heads,
            embed_size=self.config.stroke_embed_size,
            dropout=self.config.transformer_dropout,
            block_size=self.config.block_size,
            device=self.config.device,
        ).to(self.config.device)

        self.stroke_decoder = StrokeDecoderMLP(
            input_size=self.config.stroke_embed_size,
            output_size=self.config.stroke_input_size,
        )

    def forward(self, x: torch.Tensor, classname_embedding: torch.Tensor):
        batch_size, num_strokes, stroke_input_size = x.shape
        # x is of shape: (batch_size, block_size - 1, stroke_input_size)
        encoded_strokes = self.stroke_encoder_mlp(
            x
        )  # (batch_size, block_size - 1, self.stroke_embed_size)

        # TODO: Currently this assumes that the CLIP embedding size is the same as self.config.stroke_embed_size
        combined_input = torch.cat(
            (encoded_strokes, classname_embedding.unsqueeze(1)), dim=1
        )  # (batch_size, block_size, self.stroke_embed_size)

        pos_emb = self.position_embedding_table(
            torch.arange(num_strokes + 1).to(self.config.device)
        )
        combined_input = combined_input + pos_emb

        transformer_out = self.stroke_transformer(combined_input)

        decoded_out = self.stroke_decoder(transformer_out)

        return decoded_out[:, 1:, :]


class StrokeTransformer(nn.Module):
    # TODO: Add causal mask
    def __init__(
        self,
        num_transformer_blocks: int,
        num_attention_heads: int,
        embed_size: int,
        dropout: float,
        block_size: int,
        device: str,
    ):
        super().__init__()

        self.mask = torch.triu(
            torch.ones(block_size, block_size), diagonal=1
        )  # Upper triangular
        self.mask = self.mask.masked_fill(self.mask == 1, float("-inf")).to(device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_attention_heads,
            dim_feedforward=4 * embed_size,  # Common practice to use 4x embed size
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_blocks
        )
     
        # Layer norm before transformer
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, block_size, embed_size)
        x = self.layer_norm(x)
        return self.transformer(x, mask=self.mask)


class StrokeEncoderMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, block_size: int, device: str):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.device = device

        # TODO: Batchnorm, or some type of norm?
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size),
        )

    def forward(self, x):
        return self.net(x)


class StrokeDecoderMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_size),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=-1)
def calculate_loss(
    predicted: torch.Tensor,
    y: torch.Tensor,
    position_coeff: float,
    pen_state_coeff: float,
):
    assert predicted.shape == y.shape
    predicted_delta_positions = predicted[:, :, :2]
    actual_delta_positions = y[:, :, :2]
    position_loss = mse_loss(predicted_delta_positions, actual_delta_positions)

    predicted_pen_state = softmax(predicted[:, :, 2:])
    actual_pen_state = y[:, :, 2:]
    pen_state_loss = bce_loss(predicted_pen_state, actual_pen_state)

    scaled_position_loss = position_coeff * position_loss
    scaled_pen_state_loss = pen_state_coeff * pen_state_loss

    loss = scaled_position_loss + scaled_pen_state_loss
    return loss
