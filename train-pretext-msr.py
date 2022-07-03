from datasets.msr import MSRAction3D
import torch.utils.data
import torch
import time
import torchvision
from pretext import P4mask
import os
import sys
import numpy as np
import datetime
from scheduler import WarmupMultiStepLR
from utils import mkdir

# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4mask Model Training')

    parser.add_argument('--data-path', default='/home/yuhao/4DMAE/processed_data', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.7, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # encoder&decoder
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    parser.add_argument('--mask-ratio', default=0.75, type=float, help='transformer mask ratio')
    parser.add_argument('--transdim', default=384, type=int, help='transformer decoder dim')
    parser.add_argument('--decoder-depth', default=4, type=int, help='transformer decoder depth')
    parser.add_argument('--drop-path-rate', default=0.1, type=int, help='drop path rate for decoder')
    # training
    parser.add_argument('-b', '--batch-size', default=14, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--output-dir', default='/home/yuhao/P4mask/out', type=str, help='path where to save')

    args = parser.parse_args()

    return args

def main(args):

    if args.output_dir:
        mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    dataset = MSRAction3D(
            root=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=1,
            num_points=args.num_points,
            train=True
    )

    dataset_test = MSRAction3D(
            root=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=1,
            num_points=args.num_points,
            train=False
    )

    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = P4mask(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, encoder_depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, mask_ratio=args.mask_ratio, transdim=args.transdim, decoder_depth=args.decoder_depth, 
                  drop_path_rate=args.drop_path_rate)
    model.to(device)
    
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_ = 0
        b = 0
        for batch_idx, (clip, label, _) in enumerate(data_loader_train):
            clip = clip.to(device)
            print("clip:",clip.shape)
            loss = model(clip)
            loss_ += loss
            b = batch_idx
            if (batch_idx+1)%50 ==0:
                print(epoch, batch_idx, loss_/(batch_idx+1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()    ###WarmUp scheduler每一个batch都更新！！！
            sys.stdout.flush()   #############???????????
        print(epoch, loss_/(b+1))
        if (epoch+1)%5==0:
            torch.save(model, args.output_dir + "/ckpt_{}.pt".format(epoch + 1))
            
        with torch.no_grad():
            model.eval()
            loss_ = 0
            b = 0
            for batch_idx, (clip, label, _) in enumerate(data_loader_val):
                clip = clip.to(device)
                loss = model(clip)
                loss_ += loss
                b = batch_idx
                if (batch_idx+1)%50 ==0:
                    print('validation:',epoch, batch_idx, loss_/(batch_idx+1))
            print('validation:',epoch, loss_/(b+1))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = parse_args()
    main(args)