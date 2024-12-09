""" 
Adopted from "Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids" by Paschalidou et al.
https://github.com/paschalidoud/superquadric_parsing/blob/master/learnable_primitives/models.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


class NetworkParameters:
    def __init__(self, architecture, n_primitives=32,
                 mu=0.0, sigma=0.001, add_gaussian_noise=False,
                 use_sq=False, make_dense=False,
                 use_deformations=False,
                 train_with_bernoulli=False):
        self.architecture = architecture
        self.n_primitives = n_primitives
        self.train_with_bernoulli = train_with_bernoulli
        self.add_gaussian_noise = add_gaussian_noise
        self.gaussian_noise_layer = get_gaussian_noise_layer(
            self.add_gaussian_noise,
            mu,
            sigma
        )
        self.use_sq = use_sq
        self.use_deformations = use_deformations
        self.make_dense = make_dense

    @classmethod
    def from_options(cls, argument_parser):
        # Make Namespace to dictionary to be able to use it
        args = vars(argument_parser)

        architecture = args["architecture"]
        n_primitives = args.get("n_primitives", 32)

        add_gaussian_noise = args.get("add_gaussian_noise", False)
        mu = args.get("mu", 0.0)
        sigma = args.get("sigma", 0.001)

        # By default train without learning Bernoulli priors
        train_with_bernoulli = args.get("train_with_bernoulli", False)
        use_sq = args.get("use_sq", False)
        use_deformations = args.get("use_deformations", False)
        make_dense = args.get("make_dense", False)

        return cls(
            architecture,
            n_primitives=n_primitives,
            mu=mu,
            sigma=sigma,
            add_gaussian_noise=add_gaussian_noise,
            use_sq=use_sq,
            use_deformations=use_deformations,
            train_with_bernoulli=train_with_bernoulli,
            make_dense=make_dense
        )

    @property
    def network(self):
        networks = dict(
            octnet=OctnetNetwork
        )

        return networks[self.architecture.lower()]

    def primitive_layer(self, n_primitives, input_channels):
        modules = self._build_modules(n_primitives, input_channels)
        module = GeometricPrimitive(n_primitives, modules)
        return module

    def _build_modules(self, n_primitives, input_channels):
        modules = {
            "translations": Translation(n_primitives, input_channels, self.make_dense),
            "rotations": Rotation(n_primitives, input_channels, self.make_dense),
            "sizes": Size(n_primitives, input_channels, self.make_dense)
        }
        if self.train_with_bernoulli:
            modules["probs"] = Probability(n_primitives, input_channels, self.make_dense)
        if self.use_sq and not self.use_deformations:
            modules["shapes"] = Shape(n_primitives, input_channels, self.make_dense)
        if self.use_sq and self.use_deformations:
            modules["shapes"] = Shape(n_primitives, input_channels, self.make_dense)
            modules["deformations"] = Deformation(
                n_primitives, input_channels, self.make_dense)

        return modules

# Modified to accept 64x64x64 binary occupancy grid
class OctnetNetwork(nn.Module):
    def __init__(self, network_params):
        super(OctnetNetwork, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64x64x64 -> 32x32x32

            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32x32x32 -> 16x16x16

            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16x16x16 -> 8x8x8

            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8x8x8 -> 4x4x4
        )
        
        # Adjust fully connected layer based on new dimensions
        # After pooling, the size will be 4x4x4 and 64 channels
        self.encoder_fc = nn.Sequential(
            nn.Linear(4 * 4 * 4 * 64, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()
        )
        
        # Assuming the network_params contains the `primitive_layer` and `n_primitives`
        self.primitive_layer = network_params.primitive_layer(
            network_params.n_primitives,
            1024
        )

    def forward(self, X):
        # Apply convolutional layers
        X = self.encoder_conv(X)

        # Flatten the output and pass through fully connected layers
        X = X.view(-1, 4 * 4 * 4 * 64)
        X = self.encoder_fc(X)
        
        # Pass through the primitive layer
        return self.primitive_layer(X.view(-1, 1024, 1, 1, 1))

class Translation(nn.Module):
    """A layer that predicts the translation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Translation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the translation vector of each primitive, namely
        # BxMx3
        self._translation_layer = nn.Conv3d(
            input_channels, self._n_primitives*3, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the BxM*3 translation vectors for every primitive and ensure
        # that they lie inside the unit cube
        translations = torch.tanh(self._translation_layer(X)) * 0.51

        return translations[:, :, 0, 0, 0]


class Rotation(nn.Module):
    """A layer that predicts the rotation vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Rotation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the 4 quaternions of each primitive, namely
        # BxMx4
        self._rotation_layer = nn.Conv3d(
            input_channels, self._n_primitives*4, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the 4 parameters of the quaternion for every primitive
        # and add a non-linearity as L2-normalization to enforce the unit
        # norm constrain
        quats = self._rotation_layer(X)[:, :, 0, 0, 0]
        quats = quats.view(-1, self._n_primitives, 4)
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        rotations = rotations.view(-1, self._n_primitives*4)

        return rotations


class Size(nn.Module):
    """A layer that predicts the size vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Size, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the size of each primitive, along each axis,
        # namely BxMx3.
        self._size_layer = nn.Conv3d(
            input_channels, self._n_primitives*3, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Bound the sizes so that they won't take values larger than 0.51 and
        # smaller than 1e-2 (to avoid numerical instabilities with the
        # inside-outside function)
        sizes = torch.sigmoid(self._size_layer(X)) * 0.5 + 0.03
        sizes = sizes[:, :, 0, 0, 0]

        return sizes


class Shape(nn.Module):
    """A layer that predicts the shape vector
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Shape, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the shape of each primitive, along each axis,
        # namely BxMx3.
        self._shape_layer = nn.Conv3d(
            input_channels, self._n_primitives*2, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Bound the predicted shapes to avoid numerical instabilities with
        # the inside-outside function
        shapes = torch.sigmoid(self._shape_layer(X))*1.1 + 0.4
        shapes = shapes[:, :, 0, 0, 0]

        return shapes


class Deformation(nn.Module):
    """A layer that predicts the deformations
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Deformation, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the tapering parameters of each primitive.
        self._tapering_layer =\
            nn.Conv3d(input_channels, self._n_primitives*2, 1)

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # The tapering parameters are from -1 to 1
        taperings = torch.tanh(self._tapering_layer(X))*0.9
        taperings = taperings[:, :, 0, 0, 0]

        return taperings


class Probability(nn.Module):
    """A layer that predicts the probabilities
    """
    def __init__(self, n_primitives, input_channels, make_dense=False):
        super(Probability, self).__init__()
        self._n_primitives = n_primitives

        self._make_dense = make_dense
        if self._make_dense:
            self._fc = nn.Conv3d(input_channels, input_channels, 1)
            self._nonlin = nn.LeakyReLU(0.2, True)

        # Layer used to infer the probability of existence for the M
        # primitives, namely BxM numbers, where B is the batch size
        self._probability_layer = nn.Conv3d(
            input_channels, self._n_primitives, 1
        )

    def forward(self, X):
        if self._make_dense:
            X = self._nonlin(self._fc(X))

        # Compute the BxM probabilities of existence for the M primitives and
        # remove unwanted axis with size 1
        probs = torch.sigmoid(
           self._probability_layer(X)
        ).view(-1, self._n_primitives)

        return probs


class PrimitiveParameters(object):
    """Represents the \lambda_m."""
    def __init__(self, probs, translations, rotations, sizes, shapes,
                 deformations):
        self.probs = probs
        self.translations = translations
        self.rotations = rotations
        self.sizes = sizes
        self.shapes = shapes
        self.deformations = deformations

        # Check that everything has a len(shape) > 1
        for x in self.members[:-2]:
            assert len(x.shape) > 1

    def __getattr__(self, name):
        if not name.endswith("_r"):
            raise AttributeError()

        prop = getattr(self, name[:-2])
        if not torch.is_tensor(prop):
            raise AttributeError()

        return prop.view(self.batch_size, self.n_primitives, -1)

    @property
    def members(self):
        return (
            self.probs,
            self.translations,
            self.rotations,
            self.sizes,
            self.shapes,
            self.deformations
        )

    @property
    def batch_size(self):
        return self.probs.shape[0]

    @property
    def n_primitives(self):
        return self.probs.shape[1]

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        return self.members[i]


class GeometricPrimitive(nn.Module):
    def __init__(self, n_primitives, primitive_params):
        super(GeometricPrimitive, self).__init__()
        self._n_primitives = n_primitives
        self._primitive_params = primitive_params

        self._update_params()

    def _update_params(self):
        for i, m in enumerate(self._primitive_params.values()):
            self.add_module("layer%d" % (i,), m)

    def forward(self, X):
        if "probs" not in self._primitive_params.keys():
            probs = X.new_ones((X.shape[0], self._n_primitives))
        else:
            probs = self._primitive_params["probs"].forward(X)

        translations = self._primitive_params["translations"].forward(X)
        rotations = self._primitive_params["rotations"].forward(X)
        sizes = self._primitive_params["sizes"].forward(X)

        # By default the geometric primitive is a cuboid
        if "shapes" not in self._primitive_params.keys():
            shapes = X.new_ones((X.shape[0], self._n_primitives*2)) * 0.25
        else:
            shapes = self._primitive_params["shapes"].forward(X)

        if "deformations" not in self._primitive_params.keys():
            deformations = X.new_zeros((X.shape[0], self._n_primitives*2))
        else:
            deformations = self._primitive_params["deformations"].forward(X)

        return PrimitiveParameters(
            probs, translations, rotations, sizes,
            shapes, deformations
        )


class GaussianNoise(nn.Module):
    def __init__(self, mu=0.0, sigma=0.01):
        super(GaussianNoise, self).__init__()
        # Mean of the distribution
        self.mu = mu
        # Standard deviation of the distribution
        self.sigma = sigma

    def forward(self, X):
        if self.training and self.sigma != 0:
            n = X.new_zeros(*X.size()).normal_(self.mu, self.sigma)
            X = X + n
        return X


def train_on_batch(
    model,
    optimizer,
    loss_fn,
    X,
    y_target,
    regularizer_terms,
    sq_sampler,
    loss_options
):
    # Zero the gradient's buffer
    optimizer.zero_grad()
    y_hat = model(X)
    loss, debug_stats = loss_fn(
        y_hat,
        y_target,
        regularizer_terms,
        sq_sampler,
        loss_options
    )
    # Do the backpropagation
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Do the update
    optimizer.step()

    return (
        loss.item(),
        [x.data if hasattr(x, "data") else x for x in y_hat],
        debug_stats
    )


def get_gaussian_noise_layer(add_gaussian_noise, mu=0.0, sigma=0.01):
    if add_gaussian_noise:
        return GaussianNoise(mu=mu, sigma=sigma)
    else:
        return GaussianNoise(mu=0.0, sigma=0.0)


def optimizer_factory(args, model):
    """Based on the input arguments create a suitable optimizer object
    """
    if args.probs_only:
        params = model._primitive_layer._primitive_params["probs"].parameters()
    else:
        params = model.parameters()

    if args.optimizer == "SGD":
        return optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        return optim.Adam(
            params,
            lr=args.lr
        )