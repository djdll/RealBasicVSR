import argparse
import glob
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model

VIDEO_EXTENSIONS = ('.mp4', '.mov')
#python inference_realbasicvsr.py configs/realbasicvsr_x4.py checkpoints/RealBasicVSR_x4.pth data/demo_001.mp4 results/demo_001.mp4 --fps=12.5
#python inference_realbasicvsr.py configs/realbasicvsr_x4.py checkpoints/RealBasicVSR_x4.pth data/demo_000 results/demo_000

def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg) # vgg19
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


def main():
    #args = parse_args()
    config_test = 'configs/realbasicvsr_x4.py'
    checkpoint = 'checkpoints\RealBasicVSR_x4.pth'
    input_dir = 'data/demo_000'  # Img Path
    output_dir = 'results/demo'
    max_seq_len = None
    is_save_as_png = True
    #fps = 15
    # initialize the model
    model = init_model(config_test, checkpoint)

    # read images
    file_extension = os.path.splitext(input_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:  # input is a video file
        video_reader = mmcv.VideoReader(input_dir)
        inputs = []
        for frame in video_reader:
            inputs.append(np.flip(frame, axis=2))
    elif file_extension == '':  # input is a directory
        inputs = []
        input_paths = sorted(glob.glob(f'{input_dir}/*'))
        for input_path in input_paths:
            img = mmcv.imread(input_path, channel_order='rgb')
            inputs.append(img)
    else:
        raise ValueError('"input_dir" can only be a video or a directory.')

    for i, img in enumerate(inputs):
        img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        inputs[i] = img.unsqueeze(0)
    inputs = torch.stack(inputs, dim=1)

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True

    with torch.no_grad():
        if isinstance(max_seq_len, int):
            outputs = []
            for i in range(0, inputs.size(1), max_seq_len):
                imgs = inputs[:, i:i + max_seq_len, :, :, :]
                if cuda_flag:
                    imgs = imgs.cuda()
                outputs.append(model(imgs, test_mode=True)['output'].cpu())
            outputs = torch.cat(outputs, dim=1)
        else:
            if cuda_flag:
                inputs = inputs.cuda()
            outputs = model(inputs, test_mode=True)['output'].cpu()

    if os.path.splitext(output_dir)[1] in VIDEO_EXTENSIONS:
        output_dir = os.path.dirname(output_dir)
        mmcv.mkdir_or_exist(output_dir)

        h, w = outputs.shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_dir, fourcc, fps,
                                       (w, h))
        for i in range(0, outputs.size(1)):
            img = tensor2img(outputs[:, i, :, :, :])
            video_writer.write(img.astype(np.uint8))
        cv2.destroyAllWindows()
        video_writer.release()
    else:
        mmcv.mkdir_or_exist(output_dir)
        for i in range(0, outputs.size(1)):
            output = tensor2img(outputs[:, i, :, :, :])
            filename = os.path.basename(input_paths[i])
            if is_save_as_png:
                file_extension = os.path.splitext(filename)[1]
                filename = filename.replace(file_extension, '.png')
            mmcv.imwrite(output, f'{output_dir}/{filename}')


if __name__ == '__main__':
    main()
