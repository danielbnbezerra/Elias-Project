import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
output_size = 10 # Qtd de variáveis de saída
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28 #Qtd de variáveis de entrada
sequence_length = 28 #Tamanho da janela
hidden_size = 128 #Qtd de janelas passadas
num_layers = 2

# Fully connected neural network with one hidden layer
class LHCModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LHCModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # -> x needs to be: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate lstm
        out, _ = self.lstm(x, (h0,c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out


#APENAS DENTRO DA JANELA QUANDO FOR TREINAR

# model = LHCModel(input_size, hidden_size, num_layers, output_size).to(device)
#
# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Train the model
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [N, 1, 28, 28]
#         # resized: [N, 28, 28]
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
#
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')