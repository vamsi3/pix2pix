import argparse
import csv
import os
import time
import torch
import torchvision

from data import dataloader
from model import Pix2Pix


EXECUTION_ID = time.strftime('%m_%d_%H_%M_%S')

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data_root', required=True, type=str)
parser.add_argument('--data_resize', default=286, type=int)
parser.add_argument('--data_crop', default=256, type=int)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--data_invert', action='store_true')
parser.add_argument('--out_root', default=os.path.join('.', 'output'), type=str)

# Training
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--val_num_batches', default=5, type=int)
parser.add_argument('--pretrain_timestamp', type=str)
parser.add_argument('--save_model_rate', default=1, type=int)

# Optimization
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lambda_l1', default=100.0, type=float)
parser.add_argument('--lambda_d', default=0.5, type=float)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_root) if not os.path.exists(args.out_root) else None

    if args.mode == 'train':
        out_root = os.path.join(args.out_root, f"{EXECUTION_ID}")
        out_image_path = os.path.join(out_root, 'images')
        os.makedirs(out_image_path)

        with open(f'{out_root}/config.txt', 'w') as config:
            config.write(args.__str__())

        train_dataloader = dataloader(os.path.join(args.data_root, 'train'), args.dataset, invert=args.data_invert, train=True, shuffle=True,
                                           device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop)
        val_dataloader = dataloader(os.path.join(args.data_root, 'val'), args.dataset, invert=args.data_invert, train=False, shuffle=True,
                                         device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop)

    elif args.mode == 'test':
        out_root = os.path.join(args.out_root, args.pretrain_timestamp)
        out_image_path = os.path.join(
            out_root, 'images', f"test_{EXECUTION_ID}")
        os.makedirs(out_image_path)

        test_dataloader = dataloader(os.path.join(args.data_root, 'test'), args.dataset, invert=args.data_invert, train=False, shuffle=True,
                                          device=device, batch_size=args.batch_size, resize=args.data_resize, crop=args.data_crop)

    model = Pix2Pix(lr=args.lr, lambda_l1=args.lambda_l1, lambda_d=args.lambda_d, dataset=args.dataset)

    start_epoch = 0
    if args.pretrain_timestamp:
        checkpoint_dir = os.path.join(args.out_root, args.pretrain_timestamp)
        for filename in reversed(sorted(os.listdir(checkpoint_dir))):
            if filename.startswith('epoch_'):
                break
        print(f"Using pretrained model... {args.pretrain_timestamp}/{filename}")
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch'] + 1

    model.to(device)

    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch + args.num_epochs):
            train_loss_file = open(f'{out_root}/train_loss.csv', 'a', newline='')
            val_loss_file = open(f'{out_root}/val_loss.csv', 'a', newline='')
            train_loss_writer = csv.writer(train_loss_file, delimiter=',')
            val_loss_writer = csv.writer(val_loss_file, delimiter=',')

            print(f"\n------------ Epoch {epoch} ------------")
            model.scheduler_step()

            clock_tick = time.time()
            for batch_index, data in enumerate(train_dataloader):
                loss, output_g = model.train(data)
                if batch_index % 100 == 0:
                    stats_string = ''.join(f" | {k} = {v:6.3f}" for k, v in loss.items())
                    print(f"[TRAIN]  batch_index = {batch_index:03d} {stats_string}")
                    train_loss_writer.writerow([epoch + 1, batch_index, stats_string])

                    if args.dataset == 'places':
                        data = list(data)
                        data[0] = data[0].repeat(1, 3, 1, 1)
                        data = tuple(data)

                    image_tensor = torch.cat((*data, output_g), dim=3)
                    torchvision.utils.save_image(image_tensor, os.path.join(
                        out_image_path, f"train_{epoch}_{batch_index}.png"), nrow=1, normalize=True)
            clock_tok = time.time()
            print(f"[CLOCK] Time taken: {(clock_tok - clock_tick) / 60 : .3f} minutes")

            for batch_index, data in enumerate(val_dataloader):
                if batch_index >= args.val_num_batches:
                    break
                loss, output_g = model.eval(data)
                stats_string = ''.join(f" | {k} = {v:6.3f}" for k, v in loss.items())
                print(f"[VAL]    batch_index = {batch_index:03d} {stats_string}")
                val_loss_writer.writerow([epoch + 1, batch_index, stats_string])

                if args.dataset == 'places':
                    data = list(data)
                    data[0] = data[0].repeat(1, 3, 1, 1)
                    data = tuple(data)

                image_tensor = torch.cat((*data, output_g), dim=3)
                torchvision.utils.save_image(image_tensor, os.path.join(
                    out_image_path, f"val_{epoch}_{batch_index}.png"), nrow=1, normalize=True)

            if epoch % args.save_model_rate == 0:
                checkpoint_file_path = os.path.join(
                    out_root, f"epoch_{epoch}.pt")
                torch.save({
                    'state': model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_file_path)

            train_loss_file.close()
            val_loss_file.close()

    elif args.mode == 'test':
        for batch_index, data in enumerate(test_dataloader):
            loss, output_g = model.eval(data)

            if args.dataset == 'places':
                data = list(data)
                data[0] = data[0].repeat(1, 3, 1, 1)
                data = tuple(data)

            image_tensor = torch.cat((*data, output_g), dim=3)
            torchvision.utils.save_image(image_tensor, os.path.join(
                out_image_path, f"{batch_index * args.batch_size}_{batch_index * args.batch_size - 1}.png", nrow=1, normalize=True))
