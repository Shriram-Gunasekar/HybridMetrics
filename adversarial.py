import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchmetrics import FID

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, img_shape[0] * img_shape[1] * img_shape[2]),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(img_shape[0] * img_shape[1] * img_shape[2], 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Hyperparameters
latent_dim = 100
img_shape = (1, 28, 28)
num_epochs = 50
batch_size = 64
lr = 0.0002

# Initialize networks
generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)

# Loss function and optimizer
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Dummy data (real images)
def generate_real_samples(batch_size):
    return torch.randn(batch_size, *img_shape)

# Training loop
fid_metric = FID()
for epoch in range(num_epochs):
    for i in range(1000):  # Number of batches per epoch (adjust as needed)
        # Generate real and fake samples
        real_images = generate_real_samples(batch_size)
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_outputs = discriminator(real_images)
        d_loss_real = adversarial_loss(real_outputs, real_labels)

        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)

        g_loss = adversarial_loss(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Update FID metric
        fid_metric(fake_images, real_images)

    # Print FID score every few epochs
    if (epoch + 1) % 5 == 0:
        fid_score = fid_metric.compute()
        print(f"Epoch [{epoch+1}/{num_epochs}], FID: {fid_score}")

# Save generated images
z = torch.randn(64, latent_dim)
fake_images = generator(z)
save_image(fake_images, 'generated_images.png', nrow=8, normalize=True)
