"""A/B/C 共用模型工厂；模型差异由 JSON 配置定义。"""
import torch
from torch import nn


def validate_config(config):
    for key in ('encoder_depth', 'encoder_dim', 'encoder_heads', 'predictor_depth',
                'predictor_dim', 'predictor_heads', 'mlp_ratio', 'patch_size', 'image_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for part in ('encoder', 'predictor'):
        if config[f'{part}_dim'] % config[f'{part}_heads']:
            raise ValueError(f'{part} width must be divisible by heads')
    if config['image_size'] != 32 or config['image_size'] % config['patch_size']:
        raise ValueError('This CIFAR-10 runner requires image_size=32 and a dividing patch_size')
    if config['position_encoding'] not in ('learned', 'fixed_2d_sincos'):
        raise ValueError('Unsupported position_encoding')
    if config['predictor_style'] not in ('baseline', 'bottleneck'):
        raise ValueError('Unsupported predictor_style')
    if config['predictor_style'] == 'baseline' and config['predictor_dim'] != config['encoder_dim']:
        raise ValueError('Baseline predictor must have the same width as encoder')
    if config['position_encoding'] == 'fixed_2d_sincos' and config['encoder_dim'] % 4:
        raise ValueError('Fixed position dimensions must be divisible by 4')
    if config['predictor_style'] == 'bottleneck' and config['predictor_dim'] % 4:
        raise ValueError('Predictor position dimension must be divisible by 4')


def patchify(images, patch_size):
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError('Image dimensions must be divisible by patch_size')
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, c * patch_size**2)


def fixed_2d_sincos(grid_size, dim):
    coordinates = torch.arange(grid_size, dtype=torch.float64)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
    frequency = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float64) / (dim // 4)))
    def encode(coordinate):
        phase = coordinate.reshape(-1, 1) * frequency
        return torch.cat([phase.sin(), phase.cos()], dim=-1)
    return torch.cat([encode(xx), encode(yy)], dim=-1).float().unsqueeze(0)


def transformer(config, part):
    dim = config[f'{part}_dim']
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=config[f'{part}_heads'],
        dim_feedforward=dim * config['mlp_ratio'], dropout=config['dropout'],
        batch_first=True, activation=config['activation'], norm_first=config['norm_first'],
        layer_norm_eps=config['layer_norm_eps'],
    )
    norm = nn.LayerNorm(dim, eps=config['layer_norm_eps']) if config['final_norm'] else None
    return nn.TransformerEncoder(layer, num_layers=config[f'{part}_depth'], norm=norm,
                                 enable_nested_tensor=False)


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        self.patch_embedding = nn.Linear(3 * config['patch_size']**2, config['encoder_dim'])
        grid = config['image_size'] // config['patch_size']
        if config['position_encoding'] == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, grid**2, config['encoder_dim']) * 0.02)
        else:
            self.register_buffer('pos_embedding', fixed_2d_sincos(grid, config['encoder_dim']))
        self.encoder = transformer(config, 'encoder')

    def forward(self, patches, indices=None):
        tokens = self.patch_embedding(patches) + self.pos_embedding
        if indices is not None:
            tokens = tokens[:, indices, :]
        return self.encoder(tokens)


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_config(config)
        self.style = config['predictor_style']
        dim = config['predictor_dim']
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        if self.style == 'bottleneck':
            self.input_projection = nn.Linear(config['encoder_dim'], dim)
            grid = config['image_size'] // config['patch_size']
            self.register_buffer('pos_embedding', fixed_2d_sincos(grid, dim))
        self.transformer = transformer(config, 'predictor')
        if self.style == 'bottleneck':
            self.output_projection = nn.Linear(dim, config['encoder_dim'])

    def forward(self, context_features, context_index, target_index, encoder_positions):
        batch, n_context, _ = context_features.shape
        target = self.mask_token.expand(batch, target_index.numel(), -1)
        if self.style == 'baseline':
            context = context_features
            target = target + encoder_positions[:, target_index, :]
        else:
            context = self.input_projection(context_features) + self.pos_embedding[:, context_index, :]
            target = target + self.pos_embedding[:, target_index, :]
        output = self.transformer(torch.cat([context, target], dim=1))[:, n_context:, :]
        return self.output_projection(output) if self.style == 'bottleneck' else output


def random_block_mask(grid_size, block_size, device, generator):
    if not 0 < block_size < grid_size:
        raise ValueError('Mask must leave both context and target tokens')
    row = torch.randint(grid_size - block_size + 1, (1,), generator=generator).item()
    col = torch.randint(grid_size - block_size + 1, (1,), generator=generator).item()
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    mask[row:row+block_size, col:col+block_size] = True
    return torch.where(~mask.flatten())[0], torch.where(mask.flatten())[0]
