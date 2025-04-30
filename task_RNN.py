import numpy as np

text = "i love deep learning"
words = text.split()
vocab = list(set(words))
vocab_size = len(vocab)

word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

inputs = [one_hot(word_to_ix[w], vocab_size) for w in words[:3]]
target = word_to_ix[words[3]]

hidden_size = 8
learning_rate = 0.1

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

for epoch in range(500):
    hs = {}
    hs[-1] = np.zeros((hidden_size, 1))
    loss = 0

    for t in range(3):
        x = inputs[t].reshape(-1, 1)
        hs[t] = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh)

    y_hat = softmax(np.dot(Why, hs[2]) + by)
    target_vec = one_hot(target, vocab_size).reshape(-1, 1)
    loss = -np.sum(target_vec * np.log(y_hat))

    dWhy = np.dot((y_hat - target_vec), hs[2].T)
    dby = y_hat - target_vec

    dh = np.dot(Why.T, dby)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)

    for t in reversed(range(3)):
        dtanh = (1 - hs[t] ** 2) * dh
        x = inputs[t].reshape(-1, 1)
        dWxh += np.dot(dtanh, x.T)
        dWhh += np.dot(dtanh, hs[t-1].T)
        dbh += dtanh
        dh = np.dot(Whh.T, dtanh)

    for param, dparam in zip([Wxh, Whh, Why, bh, by],
                             [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam

    if (epoch + 1) % 100 == 0:
        pred_index = np.argmax(y_hat)
        print(epoch + 1, round(loss.item(), 4), ix_to_word[pred_index])

print("Final Prediction:", ix_to_word[np.argmax(y_hat)])
