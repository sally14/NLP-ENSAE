python train.py taoteba  logs/logs_taoteba_LSTM_Dense \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=10 \
             --batch_size=20

python train.py wiki  logs/logs_wiki_LSTM_Dense \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=10 \
             --batch_size=20

python train.py taoteba  logs/logs_taoteba_LSTM \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20

python train.py wiki  logs/logs_wiki_LSTM \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20

python train.py taoteba  logs/logs_taoteba_Encoder \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=True \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20

python train.py wiki  logs/logs_wiki_Encoder \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=True \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20

python train.py wiki  logs/logs_taoteba_Encoder_no_chars \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=False \
             --weighted_loss=False \
             --add_encoder=True \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20

python train.py taoteba  logs/logs_taoteba_LSTM_chars_weighted_loss \
             --optimizer=adam \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=True \
             --add_encoder=True \
             --deepness_finish=0 \
             --n_epochs=10 \
             --batch_size=20



