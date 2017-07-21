# Program of recommending movies
import math
import os

import numpy as np
import tensorflow as tf

from com.mbot.recommender.input_reader import read_input_data, read_mvoies_meta_data


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


def cosine_similarity_embeddings(embeddings, valid_dataset):
    """Compute the cosine similarity between all movies embeddings."""
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    return similarity


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(low_dim_embs)

        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(filename)
    except ImportError:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')


def main():
    input_data = read_input_data()
    movie_meta = read_mvoies_meta_data('../../../data/ml-100k/u.item')

    # Variables and vectors for calculation
    rating_mat_data = input_data.rating_mat
    rating_exist_data = input_data.is_rated_mat

    # Print the sparsity in the input data
    print_sparsity(rating_mat_data)

    # normalize the input values
    normalize_ratings(rating_mat_data, rating_exist_data)

    learning_rate = 0.001

    rating_mat = tf.placeholder(tf.float32, [input_data.user_cnt, input_data.movie_cnt], name="ratings")
    rating_exist = tf.placeholder(tf.float32, [input_data.user_cnt, input_data.movie_cnt])

    # Parameters to be learned
    user_features = tf.Variable(tf.random_normal([input_data.user_cnt, input_data.feature_cnt],
                                                 dtype=tf.float32), name="user_feature")  # user features to be learned
    movie_features = tf.Variable(tf.random_normal([input_data.movie_cnt, input_data.feature_cnt]),
                                 dtype=tf.float32, name="movie_feature")  # movies features to be learned

    # cost = cost_function(rating_mat, rating_exist, user_features, movie_features)
    # tf.clip_by_value(user_features, 1e-10, 1e+10), tf.clip_by_value(movie_features, 1e-10, 1e+10)

    # Clip off the values as the gradients may be vanishing (gradients and features tend to negative infinity).
    # Clipping can also be used for exploding gradients (weights are updated with infinity).
    rating_cal = tf.matmul(tf.clip_by_value(user_features, 1e-1, 1e+1, name="user_feature"),
                           tf.clip_by_value(movie_features, 1e-1, 1e+1, name="movie_feature"),
                           transpose_b=True)
    rating_pred = tf.multiply(rating_cal, rating_exist, name="predictRating")  # does element wise matrix multiplication

    # Regularization so that the learned weights will generalize for new input.
    user_regularization = tf.reduce_sum(np.square(tf.clip_by_value(user_features, 1e-1, 1e+1, name="user_feature")),
                                        name="user_reg")
    movie_regularization = tf.reduce_sum(tf.square(tf.clip_by_value(movie_features, 1e-1, 1e+1, name="movie_feature")),
                                         name="movie_reg")

    # cost function
    cost = tf.reduce_sum(tf.square(rating_mat - rating_pred), name="square_error") + (learning_rate) * (
        user_regularization + movie_regularization)

    # cost = tf.reduce_sum(tf.square(rating_mat - rating_pred))

    # cost = tf.reduce_mean(cost)

    # summary for cost to be displayed in TensorBoard
    tf.summary.scalar("cost", cost)

    # Summary - histogram for feature vectors
    tf.summary.histogram("user_feature", user_features)
    tf.summary.histogram("movie_feature", movie_features)

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope("train"):
        train = optimizer.minimize(cost, global_step=global_step)

    sess = tf.Session()
    init = tf.global_variables_initializer()

    # merge all the summary and write the summary to disk
    merged_summary = tf.summary.merge_all()

    # Writer for generating graph
    writer = tf.summary.FileWriter(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../graph/")))

    writer.add_graph(sess.graph)

    # Saver for saving the model after training
    saver = tf.train.Saver([user_features, movie_features])

    # initialize the variables
    sess.run(init)

    # Save the model
    saver.save(sess, "../../../model/svd-model", write_meta_graph=True)

    # train
    for step in range(0):
        sess.run(train, {rating_mat: rating_mat_data, rating_exist: rating_exist_data})

        if step % 1000 == 0:
            # save the model after certain iterations
            saver.save(sess, "../../../model/svd-model", global_step=global_step)

        if step % 100 == 0:
            print("step ", step, "Cost -->",
                  sess.run(cost, {rating_mat: rating_exist_data, rating_exist: rating_exist_data}))
            summary = sess.run(merged_summary, {rating_mat: rating_mat_data, rating_exist: rating_exist_data})
            writer.add_summary(summary, step)

    plot_with_labels(movie_features.eval(sess), movie_meta)

    # with tf.Session() as sess:
    #     # Restore variables from disk.
    #     print("Model restored.")
    #     saver = tf.train.import_meta_graph("../../../model/svd-model.meta")
    #
    #     saver.restore(sess, tf.train.latest_checkpoint('../../../model/'))
    #     # print the calculated ratings
    #     print(sess.run(tf.matmul(user_features, movie_features, transpose_b=True)))
    #     plot_with_labels(movie_features, movie_meta)

if __name__ == "__main__":
    main()
