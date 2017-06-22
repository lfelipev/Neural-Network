# Predicts an output value for a row given a set of weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation = activation + weights[i+1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, learning_rate, num_epochs):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(num_epochs):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] += learning_rate * error
            for i in range(len(row)-1):
                weights[i + 1] += learning_rate * error * row[i]
    return weights


#Perception algorithm
def perceptron(train, test, learning_rate, num_epoch):
    predictions = list()
    weights = train_weights(train, learning_rate, num_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions