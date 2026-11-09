"""一个入口，依次预训练 A/B/C；共享训练配置，独立初始化和保存。"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Backbone, Predictor, patchify, random_block_mask, validate_config

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def resolve_training(training):
    # 旧配置没有新字段时，仍采用原来的未归一化目标和 MSE。
    return {'normalize_targets': False, 'loss_type': 'mse', **training}


def prediction_loss(prediction, targets, loss_type):
    if loss_type == 'mse':
        return F.mse_loss(prediction, targets)
    if loss_type == 'smooth_l1':
        return F.smooth_l1_loss(prediction, targets, beta=1.0)
    raise ValueError('loss_type must be mse or smooth_l1')


def train_one(model_config, training, dataset, device, seed, destination):
    training = resolve_training(training)
    validate_training(training)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'model_config.json', model_config)
    write_json(destination / 'train_config.json', dict(training, seed=seed))
    torch.manual_seed(seed)
    # 模型初始化不影响样本顺序或 mask；同种子下 A/B/C 使用相同采样序列。
    order_rng = torch.Generator().manual_seed(seed)
    mask_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=training['batch_size'], shuffle=True,
                        num_workers=training['num_workers'], pin_memory=device.type == 'cuda',
                        generator=order_rng)
    if not len(loader):
        raise ValueError('Training dataset is empty')
    student = Backbone(model_config).to(device)
    teacher = deepcopy(student).requires_grad_(False).eval()
    predictor = Predictor(model_config).to(device)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()),
                                 lr=training['learning_rate'], weight_decay=training['weight_decay'])
    epochs = training['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - 1, 1), eta_min=training['min_learning_rate'])
    student.train()
    predictor.train()
    started = time.perf_counter()
    name = model_config['experiment']
    print(f'\nStarting {name} | seed={seed} | encoder={model_config["encoder_depth"]} '
          f'| predictor={model_config["predictor_depth"]} '
          f'| normalize_targets={training["normalize_targets"]} | loss={training["loss_type"]}', flush=True)
    with (destination / 'training.csv').open('w', newline='', encoding='utf-8') as log:
        writer = csv.DictWriter(log, fieldnames=['epoch', 'learning_rate', 'loss', 'seconds'])
        writer.writeheader()
        for epoch in range(epochs):
            total_loss = 0.0
            lr = optimizer.param_groups[0]['lr']
            epoch_started = time.perf_counter()
            for images, _ in loader:
                images = images.to(device)
                patches = patchify(images, model_config['patch_size'])
                context, target = random_block_mask(
                    model_config['image_size'] // model_config['patch_size'],
                    training['mask_block_size'], device, mask_rng)
                prediction = predictor(student(patches, context), context, target, student.pos_embedding)
                with torch.no_grad():
                    teacher_features = teacher(patches)
                    if training['normalize_targets']:
                        # 每个 token 沿特征维归一化；无可学习仿射参数，不改变 Encoder 本体。
                        teacher_features = F.layer_norm(teacher_features, (teacher_features.size(-1),), eps=1e-5)
                    targets = teacher_features[:, target, :]
                loss = prediction_loss(prediction, targets, training['loss_type'])
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: non-finite loss at epoch {epoch + 1}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for t, s in zip(teacher.parameters(), student.parameters()):
                        t.mul_(training['ema']).add_(s, alpha=1 - training['ema'])
                    # 固定位置是 buffer；保持不变。A 的可学习位置已包含在参数 EMA 中。
                total_loss += loss.item()
            average_loss = total_loss / len(loader)
            writer.writerow(dict(epoch=epoch + 1, learning_rate=lr, loss=average_loss,
                                 seconds=time.perf_counter() - epoch_started))
            log.flush()
            print(f'{name} seed={seed} | Epoch {epoch+1}/{epochs} | LR {lr:.8f} '
                  f'| Loss {average_loss:.6f}', flush=True)
            if epoch + 1 < epochs:
                scheduler.step()
    checkpoint = destination / 'teacher.pt'
    payload = {
        'model_config': dict(model_config), 'training_config': dict(training),
        'pretrain_seed': seed,
        'patch_embedding': {k: v.detach().cpu() for k, v in teacher.patch_embedding.state_dict().items()},
        'encoder': {k: v.detach().cpu() for k, v in teacher.encoder.state_dict().items()},
        'pos_embedding': teacher.pos_embedding.detach().cpu(),
    }
    temporary = checkpoint.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    result = dict(experiment=name, seed=seed, status='completed', checkpoint=str(checkpoint.resolve()),
                  final_loss=average_loss, seconds=time.perf_counter() - started,
                  trainable_parameters=sum(p.numel() for p in student.parameters())
                  + sum(p.numel() for p in predictor.parameters()))
    print(f'Saved: {checkpoint}', flush=True)
    return result


def validate_training(training):
    if type(training.get('normalize_targets', False)) is not bool:
        raise ValueError('normalize_targets must be true or false')
    if training.get('loss_type', 'mse') not in ('mse', 'smooth_l1'):
        raise ValueError('loss_type must be mse or smooth_l1')
    for key in ('epochs', 'batch_size', 'mask_block_size'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(training['num_workers']) is not int or training['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if not 0 <= training['ema'] < 1:
        raise ValueError('ema must be in [0, 1)')
    if not 0 <= training['min_learning_rate'] <= training['learning_rate'] or training['learning_rate'] <= 0:
        raise ValueError('Learning rates must satisfy 0 <= min <= initial and initial > 0')
    if training['weight_decay'] < 0:
        raise ValueError('weight_decay must be nonnegative')
    seeds = training['seeds']
    if not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct integers in [0, 2**63)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    parser.add_argument('--train-config', type=Path, help='Default: config-dir/train.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seeds', nargs='+', type=int, help='Override seeds for all experiments')
    parser.add_argument('--epochs', type=int, help='Override epochs for all experiments')
    args = parser.parse_args(argv)
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Experiments must be distinct')
    training = resolve_training(json.loads(
        (args.train_config or args.config_dir / 'train.json').read_text(encoding='utf-8')))
    if args.seeds is not None:
        training['seeds'] = args.seeds
    if args.epochs is not None:
        training['epochs'] = args.epochs
    configs = []
    try:
        validate_training(training)
        for name in args.experiments:
            config = json.loads((args.config_dir / f'{name.lower()}.json').read_text(encoding='utf-8'))
            validate_config(config)
            if config['experiment'] != name:
                raise ValueError(f'{name}: experiment name does not match config')
            if not 0 < training['mask_block_size'] < config['image_size'] // config['patch_size']:
                raise ValueError(f'{name}: invalid mask size')
            configs.append(config)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    device = torch.device(device_name)
    dataset = datasets.CIFAR10(str(args.data_dir), train=True, download=True, transform=transforms.ToTensor())
    run_dir = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(device=str(device), torch_version=str(torch.__version__), training_config=training,
                   sampling='independent generators for batch order and masks', results=[])
    write_json(run_dir / 'summary.json', summary)
    print(f'Device: {device}\nResults: {run_dir.resolve()}', flush=True)
    for seed in training['seeds']:
        for config in configs:
            record = dict(experiment=config['experiment'], seed=seed, status='running')
            summary['results'].append(record)
            write_json(run_dir / 'summary.json', summary)
            try:
                result = train_one(config, training, dataset, device, seed,
                                   run_dir / config['experiment'] / f'seed_{seed}')
                record.update(result)
            except Exception as error:
                record.update(status='failed', error=str(error))
                write_json(run_dir / 'summary.json', summary)
                raise
            finally:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            write_json(run_dir / 'summary.json', summary)
    print(f'All experiments completed. Summary: {run_dir / "summary.json"}', flush=True)
    return run_dir


if __name__ == '__main__':
    main()
