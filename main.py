#!/usr/bin/env python3
"""Simple project entrypoint to run training or testing with an options file.

Usage:
  python main.py --train --config_path ./experiment.cfg
  python main.py --test  --config_path ./experiment.cfg

This script forwards to basicsr/train.py or basicsr/test.py preserving the
project's original CLI behavior.
"""
import argparse
import yaml
import subprocess
import sys
import os


def run_command(cmd):
    print('Running:', ' '.join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f'Command failed with exit code {ret}')
        sys.exit(ret)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_repo_path(repo_root, maybe_path):
    if maybe_path is None:
        return None
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(repo_root, maybe_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--nproc_per_node', type=int, default=1, help='GPUs to use for launch')
    parser.add_argument('--launcher', type=str, default='pytorch', help='launcher for distributed')
    parser.add_argument('--resume_state', type=str, default=None, help='optional path to resume state (.state) or checkpoint (.pth)')
    args, unknown = parser.parse_known_args()

    config = args.config_path
    if not os.path.exists(config):
        print('Config not found:', config)
        sys.exit(2)

    base_py = sys.executable
    if args.train and args.test:
        print('Choose either --train or --test')
        sys.exit(2)

    if args.train:
        cmd = [base_py, '-m', 'torch.distributed.launch', '--nproc_per_node', str(args.nproc_per_node), '--master_port=4321', 'basicsr/train.py', '-opt', config, '--launcher', args.launcher]
        if args.resume_state:
            cmd += ['--resume_state', args.resume_state]
        run_command(cmd)
    elif args.test:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(config)
        test_runner = cfg.get('test_runner', {}) if isinstance(cfg, dict) else {}
        test_script = test_runner.get('script', os.path.join('tools', 'test_nafnet_blind.py'))
        test_script_path = resolve_repo_path(repo_root, test_script)

        if os.path.exists(test_script_path):
            cmd = [base_py, test_script_path]
            data_root = resolve_repo_path(repo_root, test_runner.get('data_root'))
            checkpoint = resolve_repo_path(repo_root, test_runner.get('checkpoint', cfg.get('path', {}).get('pretrain_network_g')))
            save_dir = resolve_repo_path(repo_root, test_runner.get('save_dir', os.path.join('results', cfg.get('name', 'experiment'))))
            device = test_runner.get('device', 'cuda')
            test_mask_csv = resolve_repo_path(repo_root, test_runner.get('test_mask_csv'))
            image_border = str(test_runner.get('image_border', 0))
            in_chans = str(test_runner.get('in_chans', 1))
            width = str(test_runner.get('width', 64))
            enc_blk_nums = ','.join(map(str, test_runner.get('enc_blk_nums', [1, 1, 1, 28])))
            middle_blk_num = str(test_runner.get('middle_blk_num', 1))
            dec_blk_nums = ','.join(map(str, test_runner.get('dec_blk_nums', [1, 1, 1, 1])))

            required = {'data_root': data_root, 'checkpoint': checkpoint}
            missing = [k for k, v in required.items() if not v]
            if missing:
                print('Missing test_runner fields in config:', ', '.join(missing))
                sys.exit(2)

            cmd += ['--data_root', data_root,
                    '--checkpoint', checkpoint,
                    '--in_chans', in_chans,
                    '--save_dir', save_dir,
                    '--device', device,
                    '--image_border', image_border,
                    '--width', width,
                    '--enc_blk_nums', enc_blk_nums,
                    '--middle_blk_num', middle_blk_num,
                    '--dec_blk_nums', dec_blk_nums]
            if test_mask_csv:
                cmd += ['--test_mask_csv', test_mask_csv]

            run_command(cmd)
        else:
            cmd = [base_py, '-m', 'torch.distributed.launch', '--nproc_per_node', str(args.nproc_per_node), '--master_port=4321', 'basicsr/test.py', '-opt', config, '--launcher', args.launcher]
            run_command(cmd)
    else:
        print('Specify --train or --test')
        sys.exit(2)


if __name__ == '__main__':
    main()
