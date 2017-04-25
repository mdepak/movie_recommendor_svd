# Program of recommending movies
import numpy as np

from com.mbot.recommender.input_reader import read_input_data


def cost_function(rating_mat, rating_exits, user_features, movie_features, learning_rate):
    rating_cal = np.matmul(user_features, movie_features)
    rating_cal = np.multiply( rating_cal, rating_exits)  # does element wise matrix multiplication
    user_regulrization = (learning_rate /2 )* np.sum(np.square(user_features))
    movie_regularization = (learning_rate/2) * np.sum(np.square(movie_features))
    cost = np.sum(np.square(rating_cal-rating_mat)) + user_regulrization + movie_regularization
    return cost

def main():
    input = read_input_data()

    # Variables and vectors for calculation
    rating_mat = input.rating_mat
    rating_exist = input.is_rated_mat
    user_features = np.zeros([input.user_cnt, input.feature_cnt])
    movie_features = np.zeros([input.movie_cnt, input.feature_cnt])


if __name__ == "__main__":
    main()
