import tensorflow as tf
import argparse
from res_unet import res_unet
from config import Config
from train import train
from predict import predict

def parse_args(args, config):
    """Parses the arguments to the config instance"""

    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.train_file_path = args.train_file
    config.val_file_path = args.val_file
    config.num_classes = args.num_classes
    config.use_weights = args.use_weights
    config.filters = [args.filters*2**i for i in range(args.num_layers)]
    config.input_size = args.input_size
    config.mirror = args.mirror
    config.rotate = args.rotate
    config.noise = args.noise

    return config

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="Mode must be either train or predict")
    parser.add_argument('--input_size', type=int, default=500, help="Dimension of the input image, assumes square")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--filters', type=int, default=64, help="Number of filters in initial layer")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--train_file', type=str, help="Path to the training data file")
    parser.add_argument('--val_file', type=str, help="Path to the validation data file")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the encoder")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes to predict")
    parser.add_argument('--use_weights', type=bool, default=False, help="Flag indicating whether to include a weight map")
    parser.add_argument('--mirror', type=bool, default=True, help="Set to true to mirror images during training")
    parser.add_argument('--rotate', type=bool, default=True, help="Set to true to rotate images during training")
    parser.add_argument('--noise', type=bool, default=True, help="Set to true to add gaussian to images during training")
    args = parser.parse_args()

    config = Config()
    config = parse_args(args, config)

    model = res_unet(config.input_size, config.filters, config.kernel_size, 
                     config.num_channels, config.num_classes)

    print(model.summary())

    if args.mode == 'train':
        train(model, config)
    elif args.mode == 'predict':
        predict(model, config)

if __name__ == '__main__':
    run()