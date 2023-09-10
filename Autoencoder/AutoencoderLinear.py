import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pre-Process data
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.__next__()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Batch Number, 784 pixels
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),  # N,784 -> N,128
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 12),
                                     nn.LeakyReLU(),
                                     nn.Linear(12, 3))  # -> N,3

        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.LeakyReLU(),
                                     nn.Linear(12, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, 28 * 28),
                                     # We use sigmoid here since the original image as max of 1 and min of 0
                                     # Note if it was : [-1 , 1]. we use nn.Tanh instead
                                     nn.Sigmoid())  # N,3 -> N,784

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create an instance of the Autoencoder model and move it to the GPU
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop
num_epochs = 10  # You can adjust this as needed

for epoch in range(num_epochs):
    for data in data_loader:
        inputs, _ = data
        inputs = inputs.view(inputs.size(0), -1).to(device)  # Move inputs to GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'autoencoder_model.pth')

# Visualize input and reconstructed images (move data to CPU)
with torch.no_grad():
    sample = images[0].view(1, -1).to(device)  # Take the first image from your dataset and move to GPU
    reconstructed_sample = model(sample).to("cpu")  # Move the reconstructed image back to CPU

# Plot the images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(sample.view(28, 28).cpu().numpy(), cmap='gray')  # Move the sample to CPU for plotting
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_sample.view(28, 28).numpy(), cmap='gray')
plt.show()
