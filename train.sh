python train.py taoteba  logs/logs_taoteba_LSTM  --learning_rate=0.01 --add_char_emb=False --weighted_loss=False --add_encoder=False --deepness_finish=0 --n_epochs=1 --batch_size=256



python train.py taoteba  logs/logs_taoteba_weight  --learning_rate=0.1 --add_char_emb=False --weighted_loss=True --add_encoder=False --deepness_finish=0 --n_epochs=1 --batch_size=256