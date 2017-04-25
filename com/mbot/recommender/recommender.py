# Program of recommending movies
import math

import numpy as np
import tensorflow as tf

from com.mbot.recommender.input_reader import read_input_data


def print_sparsity(matrix):
    """ Prints the sparsity in the input data """
    present_cnt = 0
    [rows, columns] = matrix.shape
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] > 1:
                present_cnt += 1
    print("Data sparsity percentage -> %r" % ((present_cnt / (rows * columns)) * 100))


def normalize_ratings(rating_matrix, rating_exists):
    """
     Normalize the movie ratings
        1) Take the mean of the ratings of movies which has ratings
        2) Subtract the mean of the movie from all the ratings existing for a movie
    """
    [rows, columns] = rating_matrix.shape
    movies_mean = np.zeros([columns])
    for i in range(columns):
        rating_present_cnt = 0
        rating_sum = 0
        for j in range(rows):
            if rating_exists[j][i] == 1:
                rating_present_cnt += 1
                rating_sum += rating_matrix[j][i]
        movies_mean[i] = np.round(rating_sum / rating_present_cnt, 3)

    for i in range(columns):
        for j in range(rows):
            if rating_exists[j][i] == 1:
                rating_matrix[j][i] -= movies_mean[i]


def cosine_similarity(v1, v2):
    """
    compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    """
    return np.dot(v1, v2) / math.sqrt(np.sum(np.square(v1)), np.sum(np.square(v2)))


def main():
    input_data = read_input_data()

    # Variables and vectors for calculation
    rating_mat_data = input_data.rating_mat
    rating_exist_data = input_data.is_rated_mat

    # Print the sparsity in the input data
    print_sparsity(rating_mat_data)

    # normalize the input values
    normalize_ratings(rating_mat_data, rating_exist_data)

    learning_rate = 0.001

    rating_mat = tf.placeholder(tf.float32, [input_data.user_cnt, input_data.movie_cnt])
    rating_exist = tf.placeholder(tf.float32, [input_data.user_cnt, input_data.movie_cnt])

    # Parameters to be learned
    user_features = tf.Variable(tf.random_normal([input_data.user_cnt, input_data.feature_cnt],
                                                 dtype=tf.float32))  # user features to be learned
    movie_features = tf.Variable(tf.random_normal([input_data.movie_cnt, input_data.feature_cnt]),
                                 dtype=tf.float32)  # movies features to be learned

    # cost = cost_function(rating_mat, rating_exist, user_features, movie_features)
    # tf.clip_by_value(user_features, 1e-10, 1e+10), tf.clip_by_value(movie_features, 1e-10, 1e+10)

    # Clip off the values as the gradients may be vanishing (gradients and features tend to negative infinity).
    # Clipping can also be used for exploding gradients (weights are updated with infinity).
    rating_cal = tf.matmul(tf.clip_by_value(user_features, 1e-1, 1e+1), tf.clip_by_value(movie_features, 1e-1, 1e+1),
                           transpose_b=True)
    rating_pred = tf.multiply(rating_cal, rating_exist)  # does element wise matrix multiplication

    # Regularization so that the learned weights will generalize for new input.
    user_regularization = tf.reduce_sum(np.square(tf.clip_by_value(user_features, 1e-1, 1e+1)))
    movie_regularization = tf.reduce_sum(tf.square(tf.clip_by_value(movie_features, 1e-1, 1e+1)))

    # cost function
    cost = tf.reduce_sum(tf.square(rating_mat - rating_pred)) + (learning_rate) * (
        user_regularization + movie_regularization)

    # cost = tf.reduce_sum(tf.square(rating_mat - rating_pred))

    # cost = tf.reduce_mean(cost)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()

    # initialize the variables
    sess.run(init)

    # train
    for step in range(10000):
        print("step ", step, "Cost -->",
              sess.run(cost, {rating_mat: rating_exist_data, rating_exist: rating_exist_data}))
        # print("step ", step, "Cost -->",
        #       sess.run(cost, {rating_mat: rating_exist_data, rating_exist: rating_exist_data}), "Movie features -->",
        #       sess.run(movie_features), "User features -->", sess.run(user_features))
        sess.run(train, {rating_mat: rating_mat_data, rating_exist: rating_exist_data})

    # print the calculated ratings
    print(sess.run(tf.matmul(user_features, movie_features, transpose_b=True)))


if __name__ == "__main__":
    main()
