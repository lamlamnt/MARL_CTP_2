import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import sys

sys.path.append("..")


class Encoder(nn.Module):
    hidden_size: int
    latent_size: int
    kernel_size: int = 4
    stride: int = 2

    @nn.compact
    def __call__(self, x):
        # 1x1 CNN layer
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(
            self.hidden_size,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        # 3 1d convolutional layers
        x = nn.Conv(
            self.hidden_size // 2,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            self.hidden_size // 4,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            self.hidden_size // 8,
            kernel_size=(self.kernel_size,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)

        # Flatten first
        x = x.reshape(x.shape[0], -1)

        # 1 Dense layers
        x = nn.Dense(
            self.latent_size,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        return x


class Decoder(nn.Module):
    size_first_layer: int
    hidden_size: int
    output_size: tuple
    kernel_size: int = 4
    stride: int = 2

    @nn.compact
    def __call__(self, x):
        # use input size to determine this 180 number -> 12x10//4
        # 1 Dense layers
        x = nn.Dense(
            self.size_first_layer,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)

        # Turn into 2d
        x = x.reshape(x.shape[0], -1, self.hidden_size // 8)

        # 2 1d convolutional layers
        x = nn.ConvTranspose(
            self.hidden_size // 4,
            kernel_size=(self.kernel_size,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.ConvTranspose(
            self.hidden_size // 2,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.ConvTranspose(
            self.hidden_size,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)

        # Reshape/unflatten
        x = x.reshape(x.shape[0], self.output_size[1], self.output_size[2], x.shape[-1])

        # 1x1 CNN layer
        x = nn.ConvTranspose(
            self.output_size[0],
            kernel_size=(1, 1),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)

        # Transpose back to original shape
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


class Autoencoder(nn.Module):
    latent_size: int  # Dimension of the latent space
    hidden_size: int
    output_size: tuple  # Output_size the same as input_size
    kernel_size: int = 4
    stride: int = 2

    def setup(self):
        size_first_layer_decoder = (
            (self.output_size[1] * self.output_size[2]) * self.hidden_size // 8
        ) // 4
        self.encoder = Encoder(
            self.hidden_size, self.latent_size, self.kernel_size, self.stride
        )
        self.decoder = Decoder(
            size_first_layer_decoder,
            self.hidden_size,
            self.output_size,
            self.kernel_size,
            self.stride,
        )

    def __call__(self, x):
        latent_representation = self.encoder(x)  # Encode input to latent space
        reconstructed = self.decoder(
            latent_representation
        )  # Decode latent representation back to data space
        return latent_representation, reconstructed
