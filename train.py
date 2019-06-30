import torch

import imagedataloader
import imagemodel
import trainargs


def train_model(use_gpu, data_directory, arch, hidden_units, learning_rate, epochs, save_directory):
    process_device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    result_datasets, result_dataloaders = imagedataloader.create_dataloaders(
        data_directory)
    image_model = imagemodel.ImageModel()
    image_model.create_model(arch, hidden_units, output_units=102,
                             class_to_idx=result_datasets['train'].class_to_idx)
    image_model.set_optimizer(learning_rate)
    image_model.train(epochs, process_device, result_dataloaders)
    image_model.save(epochs, arch, save_directory)


if __name__ == "__main__":
    args = trainargs.initialize_arguments()
    train_model(args.use_gpu, args.data_directory, args.arch,
                args.hidden_units, args.learning_rate, args.epochs, args.save_dir)
