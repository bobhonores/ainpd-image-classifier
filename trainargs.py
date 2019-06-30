import argparse


def initialize_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action='store')
    parser.add_argument('--save_dir', action='store', default='',
                        dest='save_dir', help='Set directory to save checkpoints')
    parser.add_argument('--arch', action='store', dest='arch', choices=['resnet34', 'resnet50', 'restnet101', 'densenet161', 'densenet169', 'vgg19', 'vgg16'], 
                        default='resnet34', help='Choose the model architecture')
    parser.add_argument('--learning_rate', action='store', type=float, default=0.003,
                        dest='learning_rate', help='Set the learning rate')
    parser.add_argument('--hidden_units', nargs='+', type=int,
                        dest='hidden_units', help='Set a specific number of hidden units')
    parser.add_argument('--epochs', action='store', type=int, default=10,
                        dest='epochs', help='Set the number of epochs')
    parser.add_argument('--gpu', action='store_true', default=False,
                        dest='use_gpu', help='Establish to usage of GPU')

    return parser.parse_args()
