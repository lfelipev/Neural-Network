import dataset_utils
from perceptron import perceptron


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

filename = 'sonar.csv'
dataset = dataset_utils.load_csv(filename)

for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

str_column_to_int(dataset, len(dataset[0])-1)

num_folds = 3
learning_rate = 0.01
num_epochs = 500

scores = dataset_utils.evaluate_algorithm(dataset, perceptron, num_folds, learning_rate, num_folds)

print('Scores: %s' % scores)
print('Mean accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
