import argparse


def initialize_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store',
                        help='Path of image input file')
    parser.add_argument('checkpoint', action='store',
                        help='Path of checkpoint file')
    parser.add_argument('--top_k', action='store', type=int, default=5,
                        dest='top_k', help='Top KK most likely classes')
    parser.add_argument('--category_names', action='store', dest='category_names',
                        help='Path of file to mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False,
                        dest='use_gpu', help='Establish to usage of GPU')

    return parser.parse_args()
