import json
import torch
import imagemodel
import imageprocessor
import predictargs


def predict(use_gpu, checkpoint, input, top_k):
    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    image_model = imagemodel.ImageModel()
    image_model.load(checkpoint)
    image_input = imageprocessor.process_image(input)
    probs, classes = image_model.predict(image_input, device, top_k)
    category_names_path = args.category_names
    cat_to_name = None

    if category_names_path is not None:
        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)

    print_results(classes, probs, cat_to_name)


def print_results(classes, probs, cat_to_name):
    for result in zip(probs, classes):
        try:
            class_name = f" - Name: {cat_to_name[result[1]]}"
        except TypeError:
            class_name = ""

        print(
            f"Class: {result[1]}{class_name} - Probability: {result[0] * 100:.3f}%")


if __name__ == "__main__":
    args = predictargs.initialize_arguments()
    predict(args.use_gpu, args.checkpoint, args.input, args.top_k)
