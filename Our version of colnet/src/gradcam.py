import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np

class GradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.target_activations = None
        self.gradient = None

        # Register hooks
        self.hook_target_layer()
        self.hook_gradient()

    def hook_target_layer(self):
        def forward_hook(module, input, output):
            self.target_activations = output

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)

    def hook_gradient(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        target = output[0, class_idx]
        target.backward()

        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.target_activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = adaptive_avg_pool2d(cam, (input_image.size(-2), input_image.size(-1)))
        cam = torch.sigmoid(cam)

        return cam
