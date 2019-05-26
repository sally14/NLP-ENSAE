mkdir logs
python train.py taoteba  logs/logs_taoteba_LSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=False \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=6 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 

python train.py wiki  logs/logs_wiki_LSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=False \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=4 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 

python train.py taoteba  logs/logs_taoteba_CharLSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=6 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 

python train.py wiki  logs/logs_wiki_CharLSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=False \
             --deepness_finish=3 \
             --n_epochs=4 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 


python train.py taoteba  logs/logs_taoteba_EncoderLSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=True \
             --deepness_finish=3 \
             --n_epochs=6 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 

python train.py wiki  logs/logs_wiki_EncoderLSTM \
             --optimizer=rmsprop \
             --learning_rate=0.001 \
             --add_char_emb=True \
             --weighted_loss=False \
             --add_encoder=True \
             --deepness_finish=3 \
             --n_epochs=4 \
             --batch_size=252 \
             --add_n_grams_deps=True \
             --checkpoints=2000 


