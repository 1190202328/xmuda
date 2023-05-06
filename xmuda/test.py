#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import shutil
import time
import socket
import warnings

import torch
from tqdm import tqdm

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help='generate pseudo-labels')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    return validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, pselab_path=pselab_path)


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD

    ckpt2d_list = []
    ckpt3d_list = []
    if os.path.isdir(args.ckpt2d) and args.ckpt2d == args.ckpt3d:
        for ckpt_name in os.listdir(args.ckpt2d):
            if ckpt_name.find('model_2d') != -1 and ckpt_name.endswith('.pth'):
                ckpt2d_list.append(f'{args.ckpt2d}/{ckpt_name}')
            if ckpt_name.find('model_3d') != -1 and ckpt_name.endswith('.pth'):
                ckpt3d_list.append(f'{args.ckpt2d}/{ckpt_name}')
    elif not os.path.isdir(args.ckpt2d) and not os.path.isdir(args.ckpt2d):
        ckpt2d_list.append(args.ckpt2d)
        ckpt3d_list.append(args.ckpt3d)
    else:
        raise Exception('provided ckpt path has problem！')
    ckpt2d_list = sorted(ckpt2d_list)
    ckpt3d_list = sorted(ckpt3d_list)
    # 只验证一次的情况
    if len(ckpt2d_list) == 1 and len(ckpt3d_list) == 1:
        args.ckpt2d = ckpt2d_list[0]
        args.ckpt3d = ckpt3d_list[0]
        test(cfg, args, output_dir)
    else:
        # 验证2d
        print('*' * 25 + 'eval best 2d model' + '*' * 25)
        best_2d_mIou = 0
        best_2d_mIou_idx = 0
        for idx in tqdm(range(len(ckpt2d_list))):
            ckpt_2d = ckpt2d_list[idx]
            args.ckpt2d = ckpt_2d
            args.ckpt3d = ckpt3d_list[0]
            mIou_dict = test(cfg, args, output_dir)
            if mIou_dict['2D'] > best_2d_mIou:
                best_2d_mIou = mIou_dict['2D']
                best_2d_mIou_idx = idx
        print(f'best 2d mIou is {round(best_2d_mIou, 1)}, the best model ckpt is [{ckpt2d_list[best_2d_mIou_idx]}]')
        prefix, _ = os.path.splitext(ckpt2d_list[best_2d_mIou_idx])
        shutil.copy(ckpt2d_list[best_2d_mIou_idx], f'{prefix}_best.pth')
        # 验证3d
        print('*' * 25 + 'eval best 3d model' + '*' * 25)
        best_3d_mIou = 0
        best_3d_mIou_idx = 0
        for idx in tqdm(range(len(ckpt3d_list))):
            ckpt_3d = ckpt3d_list[idx]
            args.ckpt2d = ckpt2d_list[0]
            args.ckpt3d = ckpt_3d
            mIou_dict = test(cfg, args, output_dir)
            if mIou_dict['3D'] > best_3d_mIou:
                best_3d_mIou = mIou_dict['3D']
                best_3d_mIou_idx = idx
        print(f'best 3d mIou is {round(best_3d_mIou, 1)}, the best model ckpt is [{ckpt3d_list[best_3d_mIou_idx]}]')
        prefix, _ = os.path.splitext(ckpt3d_list[best_3d_mIou_idx])
        shutil.copy(ckpt3d_list[best_3d_mIou_idx], f'{prefix}_best.pth')
        # 验证最佳的2d+3d
        print('*' * 25 + 'eval best 2d + 3d model' + '*' * 25)
        args.ckpt2d = ckpt2d_list[best_2d_mIou_idx]
        args.ckpt3d = ckpt3d_list[best_3d_mIou_idx]
        test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
