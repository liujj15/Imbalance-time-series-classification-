import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader





class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=2,  # Updated to 2 LSTM layers
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n[-1]).squeeze()

        return h_n[-1].squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=2,  # Updated to 2 LSTM layers
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)


class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=None,
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        encoding_dim = 100
        super(LSTM_AE, self).__init__()

        if h_dims is None:
            h_dims = [input_dim]

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        return x.view(-1, seq_len, 1)


# Train the LSTM_AE
batch_size = 64
num_epochs = 10
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = LSTM_AE(X_train.shape[2], 128)
model = LSTM_AE(X_train.shape[2], 100)
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_dataset = TensorDataset(train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for data in train_dataloader:
        inputs = data[0].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate encoded features using the trained LSTM_AE


with torch.no_grad():
    model.eval()
    encoded_train = model.encoder(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    encoded_test = model.encoder(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()



print(encoded_train.shape)
print(encoded_test.shape)






#






