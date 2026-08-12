from __future__ import division
import warnings
import torch.nn as nn
from torchvision import transforms
import dataset
import math
import torch
import os

from utils import setup_seed
from config import args
import numpy as np
from Networks.DSFormer import model_DSFormer
from image import load_data

warnings.filterwarnings('ignore')

setup_seed(args.seed)


def main(args):
    if args['dataset'] == 'ShanghaiA':
        test_file = './npydata/ShanghaiA_test_224c.npy'
    elif args['dataset'] == 'ShanghaiB':
        test_file = './npydata/ShanghaiB_test_224c.npy'
    elif args['dataset'] == 'UCF_QNRF':
        test_file = './npydata/ucf_qnrf_test_224c.npy'
    elif args['dataset'] == 'NWPU':
        test_file = './npydata/nwpu_val_224c.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    print('test samples:', len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    model = model_DSFormer(pretrained=True)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    test_data = pre_data(val_list, args, train=False)

    '''inference'''
    prec1 = validate(test_data, model, args)


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1)
    model.eval()

    mae = 0.0
    mse = 0.0
    for i, (fname, img, gt_count) in enumerate(test_loader):
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            out1, _ = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 1 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


if __name__ == "__main__":
    params = vars(args)
    print(params)

    main(params)
