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

    @nn.compact
    def __call__(self, x):
        # 1x1 CNN layer
        x = nn.Conv(
            self.hidden_size,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # 2 1d convolutional layers
        x = nn.Conv(
            self.hidden_size // 2,
            kernel_size=(2,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.hidden_size // 4,
            kernel_size=(2,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)

        # 1 Dense layers
        x = nn.Dense(
            self.latent_size,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        return x


class Decoder(nn.Module):
    hidden_size: int
    output_size: tuple

    @nn.compact
    def __call__(self, x):
        # 1 Dense layers
        x = nn.Dense(
            self.hidden_size // 4,
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)

        # 2 1d convolutional layers
        x = nn.ConvTranspose(
            self.hidden_size // 2,
            kernel_size=(2,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            self.hidden_size,
            kernel_size=(2,),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)

        # Reshape/unflatten
        x = x.reshape(self.output_size)

        # 1x1 CNN layer
        x = nn.ConvTranspose(
            self.output_size,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=constant(0.0),
        )(x)
        return x


class Autoencoder(nn.Module):
    latent_size: int  # Dimension of the latent space
    hidden_size: int
    output_size: tuple  # Output_size the same as input_size

    def setup(self):
        self.encoder = Encoder(self.hidden_size, self.latent_size)
        self.decoder = Decoder(self.hidden_size, self.output_size)

    def __call__(self, x):
        z = self.encoder(x)  # Encode input to latent space
        z = self.decoder(z)  # Decode latent representation back to data space
        return z
