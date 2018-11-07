import os
import argparse
import pdb
from shutil import copy
import inception_score as ins
import image_processing as ip

def load_images(i_dir, size=128):
    # prepare_inception_data(o_dir, i_dir)
    image_list = []
    for root, dirs, files in os.walk(i_dir):
        if files:
            for f in files:
                file_dir = os.path.join(root, f)
                image_list.append(ip.load_image_inception(file_dir, size))
    print('Finished Loading Files')
    return image_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default="Data/synthetic_dataset/ds",
                        help='The root directory of the synthetic dataset')

    parser.add_argument('--image_size', type=int, default=128,
                        help='Size of the image to consider for calculating '
                             'inception score')

    args = parser.parse_args()

    imgs_list = load_images(args.data_dir, size=args.image_size)

    print('Extracting Inception Score')
    mean, std = ins.get_inception_score(imgs_list)
    print('Mean Inception Score: {}\nStandard Deviation: {}'.format(mean, std))
