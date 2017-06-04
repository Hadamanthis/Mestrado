# Exemplo do keras.io (Getting Started)

from keras.models import Sequential
from keras.layers import Dense, Activation

# Modelo da NN
model = Sequential()

# Adicionando as layers na NN
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Configuração do processo de aprendizagem
model.compile(loss='categorical_crossentropy',
				optimizer='sgd',
				metrics=['accuracy'])

# treinamento
# x_train e y_train são arrays Numpy
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Alternativamente, podemos alimentar os batches manualmente
# model.train_on_batch(x_batch, y_batch)

# Avaliando a performance em uma linha
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# ou gerar predições em novos dados
classes = model.predict(x_test, batch_size=128)

