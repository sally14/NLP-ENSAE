# Get the params
# Hardware config
gpu_train = params['gpu_train'] # --

# Learning params
batch_size = params["batch_size"] # --
dropout = params["dropout"] # --
lr = params["learning_rate"] # --
optim = params["optimizer"] # --
 
# Embedding params
vocab_size = params["vocab_size"]  # Retrieved in the training start
embedding_size = params['vocab_size'] # --

# LSTM params
num_LSTM_layers = params["num_LSTM_layers"] # --
hidden_size_LSTM = params["hidden_size_LSTM"] # --

# Char embedding parameters
add_chars_emb = params['add_char_emb'] # --
if add_chars_emb:
    num_layers_chars = params['num_layers_char'] # --
    hidden_size_chars = params['hidden_size_chars'] # --
    nb_chars = params['nb_chars'] ## MUST BE GENERATED

# TODO : positionnal embedding parameters

# Encoder parameters
num_heads = params['num_heads'] # --

# Finish parameters
deepness_finish = params['deepness_finish']
activation_finish = params['activation_finish']
inter_size = params['intern_size']

# Loss parameters
weighted_loss = params['weighted_loss']
frequencies = params['frequencies'] ### TO BE COMPTUES
## NB CORES
