"""统一线性评估：自动读取权重内模型配置，可一次评估整个训练批次。"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import Backbone, patchify

ROOT = Path(__file__).resolve().parent


def load_teacher(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = Backbone(payload['model_config'])
    model.patch_embedding.load_state_dict(payload['patch_embedding'], strict=True)
    model.encoder.load_state_dict(payload['encoder'], strict=True)
    position = payload['pos_embedding']
    if position.shape != model.pos_embedding.shape:
        raise ValueError('Position embedding shape mismatch')
    if model.config['position_encoding'] == 'fixed_2d_sincos' and not torch.allclose(
            position, model.pos_embedding, rtol=1e-6, atol=1e-7):
        raise ValueError('Fixed position embedding mismatch')
    with torch.no_grad():
        model.pos_embedding.copy_(position)
    return model.to(device).requires_grad_(False).eval(), payload['pretrain_seed']


@torch.no_grad()
def extract_features(model, dataset, device, batch_size):
    model.eval()
    features, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        tokens = patchify(images.to(device), model.config['patch_size'])
        features.append(model(tokens).mean(dim=1).cpu())
        labels.append(targets)
    return TensorDataset(torch.cat(features), torch.cat(labels))


def linear_probe(train_features, test_features, device, seed, epochs, batch_size):
    torch.manual_seed(seed)
    classifier = nn.Linear(train_features.tensors[0].shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.0)
    loader = DataLoader(train_features, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        classifier.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    correct = total = 0
    classifier.eval()
    with torch.no_grad():
        for features, labels in DataLoader(test_features, batch_size=batch_size, shuffle=False):
            predictions = classifier(features.to(device)).argmax(dim=1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--run-dir', type=Path, help='训练生成的带时间戳的目录')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44], help='Probe seeds')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128, help='Feature extraction batch size')
    parser.add_argument('--probe-batch-size', type=int, default=256)
    parser.add_argument('--mode', choices=['pretrained', 'compare'], default='compare')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.probe_batch_size) <= 0:
        parser.error('Epochs and batch sizes must be positive')
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**63 for s in args.seeds):
        parser.error('Probe seeds must be distinct integers in [0, 2**63)')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    device = torch.device(device_name)
    if args.run_dir:
        summary = json.loads((args.run_dir / 'summary.json').read_text(encoding='utf-8'))
        # 按目录重新定位，整个 runs 文件夹移动到其他机器后仍可用。
        checkpoints = [args.run_dir / item['experiment'] / f'seed_{item["seed"]}' / 'teacher.pt'
                       for item in summary['results'] if item['status'] == 'completed']
    else:
        checkpoints = [args.checkpoint]
    if not checkpoints or any(not path.is_file() for path in checkpoints):
        parser.error('No completed checkpoints found, or checkpoint files are missing')
    train_data = datasets.CIFAR10(str(args.data_dir), train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR10(str(args.data_dir), train=False, download=True, transform=transforms.ToTensor())
    results = []
    print('Probe seeds repeat the classifier; each checkpoint represents one pretraining run.')
    for path in checkpoints:
        teacher, pretrain_seed = load_teacher(path, device)
        config = teacher.config
        train_features = extract_features(teacher, train_data, device, args.batch_size)
        test_features = extract_features(teacher, test_data, device, args.batch_size)
        del teacher
        scores = []
        for seed in args.seeds:
            score = dict(probe_seed=seed, pretrained=linear_probe(
                train_features, test_features, device, seed, args.epochs, args.probe_batch_size))
            if args.mode == 'compare':
                torch.manual_seed(seed)
                random_model = Backbone(config).to(device).requires_grad_(False).eval()
                random_train = extract_features(random_model, train_data, device, args.batch_size)
                random_test = extract_features(random_model, test_data, device, args.batch_size)
                score['random'] = linear_probe(random_train, random_test, device, seed,
                                               args.epochs, args.probe_batch_size)
                del random_model, random_train, random_test
            print(f'{config["experiment"]} pretrain_seed={pretrain_seed}: {score}', flush=True)
            scores.append(score)
        modes = ['pretrained', 'random'] if args.mode == 'compare' else ['pretrained']
        result = dict(checkpoint=str(path.resolve()), experiment=config['experiment'],
                      pretrain_seed=pretrain_seed, scores=scores,
                      mean={m: mean(s[m] for s in scores) for m in modes},
                      probe_sd={m: stdev(s[m] for s in scores) if len(scores) > 1 else 0.0 for m in modes})
        results.append(result)
        print(f'Mean: {result["mean"]} | Probe SD: {result["probe_sd"]}')
    output = (args.run_dir or args.checkpoint.parent) / 'evaluation.json'
    output.write_text(json.dumps(dict(probe_epochs=args.epochs, probe_batch_size=args.probe_batch_size,
                                     results=results), indent=2), encoding='utf-8')
    print(f'Evaluation saved: {output}')
    return results


if __name__ == '__main__':
    main()
