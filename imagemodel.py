import time
import torch

from torch import nn
from torch import optim
from torchvision import models


class ImageModel():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.classifier = nn.Sequential()
        self.use_fc = True

    def create_model(self, model_arch, hidden_units=None, output_units=102, class_to_idx=None):
        self.define_arch(model_arch)
        input_units = self.get_input_units(model_arch)
        self.create_classifier(input_units, hidden_units, output_units)
        self.assign_classifier(self.classifier)
        self.model.class_to_idx = class_to_idx

    def define_arch(self, model_arch):
        self.model = getattr(models, model_arch)(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

    def create_classifier(self, input_units, hidden_units, output_units):
        if hidden_units is None:
            hidden_units = []

        layers = [input_units] + hidden_units + [output_units]
        last_layer_step = len(layers) - 1
        layer_step = 0

        for layer in zip(layers[:-1], layers[1:]):
            layer_step += 1
            self.classifier.add_module(
                f'fc{layer_step}', nn.Linear(layer[0], layer[1]))

            if layer_step == last_layer_step:
                self.classifier.add_module('logsoft', nn.LogSoftmax(dim=1))
            else:
                self.classifier.add_module(f'relu{layer_step}', nn.ReLU())
                self.classifier.add_module(
                    f'drop{layer_step}', nn.Dropout(0.2))

    def assign_classifier(self, classifier):
        try:
            getattr(self.model, 'fc')
            self.model.fc = classifier
        except AttributeError:
            self.model.classifier = classifier
            self.use_fc = False

    def get_input_units(self, model_arch):
        arch_inputs = {'resnet34': 512,
                       'resnet50': 2048,
                       'restnet101': 2048,
                       'densenet161': 2208,
                       'densenet169': 1664,
                       'vgg19': 25088,
                       'vgg16': 25088}

        if model_arch in arch_inputs:
            return arch_inputs[model_arch]
        else:
            return 1024

    def set_optimizer(self, learning_rate):
        if self.use_fc:
            self.optimizer = optim.Adam(
                self.model.fc.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(
                self.model.classifier.parameters(), lr=learning_rate)

    def train(self, epochs, device, dataloaders):
        start_process = time.time()
        criterion = nn.NLLLoss()
        self.model.to(device)

        for epoch in range(epochs):
            phases = ['train', 'valid']
            losses = {phase: 0 for phase in phases}
            accuracy = 0

            start = time.time()

            for phase in phases:
                steps = 0

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                with torch.set_grad_enabled(phase == 'train'):
                    for inputs, labels in dataloaders[phase]:
                        steps += 1
                        start_step = time.time()

                        inputs, labels = inputs.to(device), labels.to(device)

                        if phase == 'train':
                            self.optimizer.zero_grad()

                        output = self.model.forward(inputs)
                        loss = criterion(output, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        losses[phase] += loss.item() * inputs.size(0)

                        if phase == 'valid':
                            percentage = torch.exp(output)
                            top_p, top_class = percentage.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)).item()
                        print(
                            f"{phase.title()} Step = {steps}; Time per step: {(time.time() - start_step):.3f} seconds")

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {losses['train'] / len(dataloaders['train']):.3f}.. "
                  f"Validation loss: {losses['valid'] / len(dataloaders['valid']):.3f}.. "
                  f"Accuracy: {accuracy / len(dataloaders['valid']):.3f}")
            print(
                f"Epoc = {epoch + 1}; Time per epoch: {(time.time() - start):.3f} seconds")

        print(f"Time per process: {(time.time() - start_process):.3f} seconds")

    def save(self, epochs, model_arch, save_directory):
        checkpoint = {
            'epochs': epochs,
            'model_arch': model_arch,
            'use_fc': self.use_fc,
            'classifier': self.model.fc if self.use_fc else self.model.classifier,
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx
        }

        torch.save(checkpoint, save_directory + 'checkpoint_app.pth')

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.define_arch(checkpoint['model_arch'])

        if checkpoint['use_fc']:
            self.model.fc = checkpoint['classifier']
        else:
            self.model.classifier = checkpoint['classifier']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']

    def predict(self, image_input, device, topk):
        image_input = image_input.to(device)

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            output = self.model.forward(image_input)

        ps = torch.exp(output)
        probs, idx = ps.topk(topk, dim=1)
        probs = probs.cpu().detach().numpy().tolist()[0]
        idx = idx.cpu().detach().numpy()[0]
        inv_dict = {value: key for key,
                    value in self.model.class_to_idx.items()}
        classes = [inv_dict[item] for item in idx]

        return probs, classes
