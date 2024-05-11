import torch.cuda
import os

from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import set_seed, MnistDataset, Trainer
from model import LeNet


def arg_parser():

    parser = ArgumentParser()

    parser.add_argument('--random_seed', help=' random seed', default=42)
    parser.add_argument('--loss_path', metavar='DIR', help='the directory for loss storage', default="loss_path")
    parser.add_argument('--model_path', metavar='DIR', help='the directory for model checkpoint storage', default='model_path')
    parser.add_argument('--batch_size', help='batch size', default=1024)

    # training parameters
    parser.add_argument('--device', help='hardware device', default='cpu')
    parser.add_argument('--lr', help='learning rate', default=1e-4)
    parser.add_argument('--auto_mixed_precision', help='do amp or not', default=False)
    parser.add_argument('--if_step_lr', help='do weight decay or not', default=True)
    parser.add_argument('--lr_change_step', help='', default=1)
    parser.add_argument('--lr_change_gamma', help='', default=0.995)
    parser.add_argument('--patience', help='patience of early stop', default=50)
    parser.add_argument('--max_epoch', help='max epoch for training', default=300)

    return parser


def main():

    parser = arg_parser()
    args = parser.parse_args()
    options = vars(args)

    set_seed(42)
    print(f"\nset seed: {options['random_seed']}")

    current_path = os.getcwd()

    loss_path = os.path.join(current_path, options['loss_path'])
    if not os.path.exists(loss_path):
        print(f'\nMaking directory for loss storage: {loss_path}')
        os.mkdir(loss_path)

    model_path = os.path.join(current_path, options['model_path'])
    if not os.path.exists(model_path):
        print(f'\nMaking directory for model checkpoint storage: {model_path}')
        os.mkdir(model_path)

    # train_df_path =
    # test_df_path =

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # TODO 测试train_dataset和val_dataset是否互补
    train_dataset = MnistDataset(root_dir='./dataset/mnist/mnist_train', transform=transform, train='train')
    eval_dataset = MnistDataset(root_dir='./dataset/mnist/mnist_train', transform=transform, train='eval')
    test_dataset = MnistDataset(root_dir='./dataset/mnist/mnist_test', transform=transform, train='test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, num_workers=8)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, num_workers=8)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, num_workers=8)

    model = LeNet()

    lenet_trainer = Trainer(options=options, model=model)
    lenet_trainer.train(options=options, train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader)

    lenet_trainer.test(weight_file_path='./model_path', test_loader=test_loader)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    torch.cuda.empty_cache()

    main()
