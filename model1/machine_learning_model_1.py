from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding 


EMBEDDING_DIM = 100

print('Build Mode ......')

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))



model.compile(loss='catagorical_crossentropy', optimizer='adam', medtrics=['accuracy'])

print('Summary of the built model .............')
print(model.summary())



num_epochs = 10 
batch_size = 128 
history = model.fit(x_train, y_train, batch_size=batch_size, epochs = num_epochs, verbose = 2, validation_split=0.2)


