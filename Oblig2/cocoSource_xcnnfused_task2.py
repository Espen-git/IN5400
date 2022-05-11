from turtle import update
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ImageCaptionModel(nn.Module):
    def __init__(self, config: dict):
        """
        This is the main module class for the image captioning network
        :param config: dictionary holding neural network configuration
        """
        super(ImageCaptionModel, self).__init__()
        # Store config values as instance variables
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']

        # Create the network layers
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.output_layer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size)
        self.nn_map_size = 512  # The output size for the image features after the processing via self.inputLayer
        self.input_layer = nn.Linear(self.number_of_cnn_features, self.hidden_state_sizes)
        self.dropout = nn.Dropout(0.25)
        self.LReLU = nn.LeakyReLU()

        self.rnn = RNN(input_size=self.embedding_size + self.nn_map_size,
                        hidden_state_size=self.hidden_state_sizes,
                        num_rnn_layers=self.num_rnn_layers,
                        cell_type=self.cell_type)

    def forward(self, cnn_features, x_tokens, is_train: bool, current_hidden_state=None) -> tuple:
        """
        :param cnn_features: Features from the CNN network, shape[batch_size, number_of_cnn_features]
        :param x_tokens: Shape[batch_size, truncated_backprop_length]
        :param is_train: A flag used to select whether or not to use estimated token as input
        :param current_hidden_state: If not None, it should be passed into the rnn module. It's shape should be
                                    [num_rnn_layers, batch_size, hidden_state_sizes].
        :return: logits of shape [batch_size, truncated_backprop_length, vocabulary_size] and new current_hidden_state
                of size [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # HINT: For task 4, you might need to do self.input_layer(torch.transpose(cnn_features, 1, 2))
        processed_cnn_features = self.LReLU(self.dropout(self.input_layer(cnn_features)))
        batch_size = x_tokens.shape[0]

        if current_hidden_state is None:
            initial_hidden_state = torch.zeros([self.num_rnn_layers, batch_size, self.hidden_state_sizes])
        else:
            initial_hidden_state = current_hidden_state

        # Call self.rnn to get the "logits" and the new hidden state
        logits, hidden_state = self.rnn(x_tokens, processed_cnn_features, initial_hidden_state, self.output_layer,
                                        self.embedding_layer, is_train)

        return logits, hidden_state

######################################################################################################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='GRU'):
        """
        :param input_size: Size of the embeddings
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        input_size_list = [GRUCell(hidden_state_size, input_size)]
        new_input_size = hidden_state_size
        input_size_list.extend([GRUCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])

        self.cells = nn.ModuleList(input_size_list)

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state.to('cuda') # [nun_rnn_layers, batch_size, hidden_state_sizes]
        input_tokens = embeddings[:,0,:]  # Should have shape (batch_size, embedding_size)
        for i in range(sequence_length):
            for j in range(self.num_rnn_layers):
                if j == 0:
                    input_first_layer = torch.cat((input_tokens, processed_cnn_features), dim=1)
                    output = self.cells[0](input_first_layer, current_hidden_state[0,:,:].clone())
                    current_hidden_state[0,:,:] = torch.unsqueeze(output, 0)

                else:
                    output = self.cells[j](current_hidden_state[(j-1),:,:].clone(), current_hidden_state[j,:,:].clone())
                    current_hidden_state[j,:,:] = torch.unsqueeze(output, 0)

            logits_i = output_layer(output)
            logits_sequence.append(logits_i)
            predictions = torch.argmax(logits_i, dim=1)

            # Get the input tokens for the next step in the sequence
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i+1, :]
                else:
                    input_tokens = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state

########################################################################################################################


class GRUCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to the GRU
        """
        super(GRUCell, self).__init__()

        self.hidden_state_size = hidden_state_size
        self.input_size = input_size
        hidden_pluss_input = hidden_state_size + input_size

        # Update gate parameters
        self.weight_u = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias_u = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Reset gate parameters
        self.weight_r = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias_r = torch.nn.Parameter(torch.zeros(1, hidden_state_size))
        # Hidden state parameters
        self.weight = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias = torch.nn.Parameter(torch.zeros(1, hidden_state_size))

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for a GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, HIDDEN_STATE_SIZE]
        :return: The updated hidden state of the GRU cell. Shape: [batch_size, HIDDEN_STATE_SIZE]
        """
                
        z_t = torch.sigmoid(torch.cat((x, hidden_state), 1).mm(self.weight_u) + self.bias_u)
        r_t = torch.sigmoid(torch.cat((x, hidden_state), 1).mm(self.weight_r) + self.bias_r)
        h_t_hat = torch.tanh(torch.cat((x, (r_t*hidden_state)), 1).mm(self.weight) + self.bias)
        
        new_hidden_state = z_t * hidden_state + (1 - z_t) * h_t_hat
        return new_hidden_state
