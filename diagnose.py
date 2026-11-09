"""用同一批真实图像诊断所有已训练 Teacher；不更新任何模型参数。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import Backbone, patchify
from evaluate import load_teacher

ROOT = Path(__file__).resolve().parent


def balanced_indices(targets, per_class, seed):
    labels = torch.as_tensor(targets)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in labels.unique(sorted=True):
        candidates = torch.where(labels == label)[0]
        if len(candidates) < per_class:
            raise ValueError(f'Class {label.item()} has fewer than {per_class} samples')
        selected.append(candidates[torch.randperm(len(candidates), generator=generator)[:per_class]])
    indices = torch.cat(selected)
    return indices[torch.randperm(len(indices), generator=generator)].tolist()


def cosine_statistics(features, labels):
    # Exact mean over distinct pairs, O(ND) memory/time; no self-pairs or sampled pairs.
    norms = features.norm(dim=1)
    valid = norms > 0
    unit = features[valid] / norms[valid, None]
    labels = labels[valid]
    n = len(unit)
    if n < 2:
        return dict(mean=None, same_class=None, different_class=None, gap=None,
                    valid_vectors=n, same_pairs=0, different_pairs=0)
    total = (unit.sum(0).square().sum() - unit.square().sum()).item()
    same_total, same_count = 0.0, 0
    for label in labels.unique():
        group = unit[labels == label]
        same_total += (group.sum(0).square().sum() - group.square().sum()).item()
        same_count += len(group) * (len(group) - 1)
    all_count = n * (n - 1)
    different_count = all_count - same_count
    same = same_total / same_count if same_count else None
    different = (total - same_total) / different_count if different_count else None
    return dict(mean=total / all_count, same_class=same, different_class=different,
                gap=same - different if same is not None and different is not None else None,
                valid_vectors=n, same_pairs=same_count // 2, different_pairs=different_count // 2)


def feature_metrics(features, labels):
    x = features.detach().cpu().double()
    labels = labels.cpu()
    if x.ndim != 2 or len(x) < 2 or len(labels) != len(x) or not torch.isfinite(x).all():
        raise ValueError('Expected finite [N,D] features and matching labels, N >= 2')
    # Center using differences for exact zero on identical rows, even with a large mean.
    centered = x - x[:1]
    centered -= centered.mean(0)
    energy = x.square().sum().item()
    centered_energy = centered.square().sum().item()
    std = (centered.square().sum(0) / (len(x) - 1)).sqrt()
    singular = torch.linalg.svdvals(centered)
    total_singular = singular.sum().item()
    variance = singular.square()
    total_variance = variance.sum().item()
    if total_singular:
        p = singular[singular > 0] / total_singular
        effective_rank = (-torch.sum(p * p.log())).exp().item()
    else:
        effective_rank = 0.0  # Explicit convention for a zero centered matrix.
    fractions = variance / total_variance if total_variance else torch.zeros_like(variance)
    within = 0.0
    for label in labels.unique():
        group = centered[labels == label]
        within += (group - group.mean(0)).square().sum().item()
    between = max(centered_energy - within, 0.0)
    raw_cos = cosine_statistics(x, labels)
    centered_cos = cosine_statistics(centered, labels)
    return {
        'num_images': len(x), 'feature_dim': x.shape[1],
        'feature_std_mean': std.mean().item(), 'feature_std_min': std.min().item(),
        'feature_std_max': std.max().item(),
        'feature_norm_mean': x.norm(dim=1).mean().item(),
        'centered_energy_fraction': centered_energy / energy if energy else None,
        'effective_rank_centered': effective_rank,
        'max_centered_rank': min(len(x) - 1, x.shape[1]),
        'participation_ratio': total_variance**2 / variance.square().sum().item() if total_variance else 0.0,
        'top1_variance_fraction': fractions[0].item() if total_variance else None,
        'top10_variance_fraction': fractions[:10].sum().item() if total_variance else None,
        'between_class_variance_fraction': between / centered_energy if centered_energy else None,
        'raw_pair_cosine': raw_cos['mean'],
        'raw_same_class_cosine': raw_cos['same_class'],
        'raw_different_class_cosine': raw_cos['different_class'],
        'centered_same_class_cosine': centered_cos['same_class'],
        'centered_different_class_cosine': centered_cos['different_class'],
        'centered_class_cosine_gap': centered_cos['gap'],
        'raw_cosine_valid_vectors': raw_cos['valid_vectors'],
        'centered_cosine_valid_vectors': centered_cos['valid_vectors'],
        'per_dimension_std': std.tolist(),
        'centered_variance_spectrum': fractions.tolist(),
    }


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    features, labels = [], []
    position_sum = position_square_sum = None
    within_image_std_sum, count = 0.0, 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        tokens = model(patchify(images.to(device), model.config['patch_size'])).cpu().double()
        if not torch.isfinite(tokens).all():
            raise ValueError('Non-finite model features')
        features.append(tokens.mean(dim=1))
        labels.append(targets)
        current_sum = tokens.sum(0)
        current_square = tokens.square().sum(0)
        position_sum = current_sum if position_sum is None else position_sum + current_sum
        position_square_sum = current_square if position_square_sum is None else position_square_sum + current_square
        within_image_std_sum += tokens.std(dim=1).mean(1).sum().item()
        count += len(tokens)
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f'  Features: {count}/{len(loader.dataset)} images', flush=True)
    if count < 2:
        raise ValueError('Need at least two images')
    # Variance across images at EACH fixed spatial position, not across all patches mixed together.
    position_variance = (position_square_sum - position_sum.square() / count) / (count - 1)
    extras = dict(same_position_across_image_token_std=position_variance.clamp_min(0).sqrt().mean().item(),
                  within_image_token_std=within_image_std_sum / count)
    return torch.cat(features), torch.cat(labels), extras


def aggregate(results):
    keys = ['feature_std_mean', 'centered_energy_fraction', 'effective_rank_centered',
            'top1_variance_fraction', 'raw_pair_cosine', 'centered_class_cosine_gap',
            'between_class_variance_fraction', 'same_position_across_image_token_std']
    output = []
    for experiment in sorted({r['experiment'] for r in results}):
        for mode in ('pretrained', 'random'):
            selected = [r for r in results if r['experiment'] == experiment and r['mode'] == mode]
            if not selected:
                continue
            entry = dict(experiment=experiment, mode=mode, runs=len(selected), metrics={})
            for key in keys:
                values = [r['metrics'][key] for r in selected if r['metrics'][key] is not None]
                entry['metrics'][key] = dict(mean=mean(values) if values else None,
                                             sd=stdev(values) if len(values) > 1 else None,
                                             valid_runs=len(values))
            output.append(entry)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--per-class', type=int, default=500, help='500 per class = 5000 training images')
    parser.add_argument('--sample-seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--skip-random', action='store_true')
    args = parser.parse_args(argv)
    if args.per_class < 2 or args.batch_size < 1 or not 0 <= args.sample_seed < 2**63:
        parser.error('per-class >= 2, batch-size >= 1, and sample-seed in [0, 2**63) are required')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    device = torch.device(device_name)
    summary = json.loads((args.run_dir / 'summary.json').read_text(encoding='utf-8'))
    completed = [r for r in summary['results'] if r['status'] == 'completed']
    if not completed:
        parser.error('No completed runs in summary.json')
    checkpoints = []
    for row in completed:
        # Whitelist names from the experiment suite; do not trust stored absolute paths.
        if row['experiment'] not in ('A', 'B', 'C') or type(row['seed']) is not int or not 0 <= row['seed'] < 2**63:
            parser.error('Invalid experiment/seed in summary')
        path = args.run_dir / row['experiment'] / f'seed_{row["seed"]}' / 'teacher.pt'
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
        checkpoints.append((row, path))
    dataset = datasets.CIFAR10(str(args.data_dir), train=True, download=True, transform=transforms.ToTensor())
    indices = balanced_indices(dataset.targets, args.per_class, args.sample_seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    output_dir = args.run_dir / ('diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(dataset='CIFAR10 train', preprocessing='ToTensor only', pooling='mean over final patch tokens',
                  sample_seed=args.sample_seed, samples_per_class=args.per_class, sample_indices=indices,
                  device=str(device), torch_version=str(torch.__version__),
                  effective_rank_definition='exp(entropy(s/s.sum())) of centered feature singular values; zero matrix -> 0',
                  notes=['Descriptive diagnostics, not automatic collapse verdicts.',
                         'No model or classifier training; test images are not used.',
                         'Random baseline is architecture-matched with seed equal to pretrain_seed.',
                         'Pooled rank can be high even when variation is tiny; inspect energy and std together.',
                         'Global centered pair cosine is omitted because centering constrains its average.'],
                  results=[])
    def save():
        report['aggregate'] = aggregate(report['results'])
        temporary = output_dir / 'diagnostics.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(output_dir / 'diagnostics.json')
    save()
    for row, path in checkpoints:
        model, seed = load_teacher(path, device)
        config = dict(model.config)
        if config['experiment'] != row['experiment'] or seed != row['seed']:
            raise ValueError(f'Checkpoint identity does not match summary: {path}')
        modes = ('pretrained',) if args.skip_random else ('pretrained', 'random')
        for mode in modes:
            if mode == 'random':
                torch.manual_seed(seed)
                model = Backbone(config).to(device).requires_grad_(False).eval()
            print(f'{config["experiment"]} seed={seed} {mode}', flush=True)
            features, labels, token_metrics = extract(model, loader, device)
            metrics = feature_metrics(features, labels)
            metrics.update(token_metrics)
            report['results'].append(dict(experiment=config['experiment'], pretrain_seed=seed,
                                          mode=mode, model_config=config,
                                          checkpoint=str(path.resolve()) if mode == 'pretrained' else None,
                                          metrics=metrics))
            print(f'  Std={metrics["feature_std_mean"]:.6f} '
                  f'| centered erank={metrics["effective_rank_centered"]:.2f} '
                  f'| raw cosine={metrics["raw_pair_cosine"]}', flush=True)
            save()
            del model, features, labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    print(f'Completed. Send back: {output_dir / "diagnostics.json"}', flush=True)
    return output_dir


if __name__ == '__main__':
    main()
