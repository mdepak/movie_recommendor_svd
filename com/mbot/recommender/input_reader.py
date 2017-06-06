import csv
import os

import numpy as np


class InputData:
    user_cnt = None
    movie_cnt = None
    feature_cnt = None
    rating_mat = None
    is_rated_mat = None


def read(file, ratings, rating_exists):
    script_path = os.path.dirname(__file__)
    file_path = os.path.join("../../../data/ml-100k", file)
    file_path = os.path.join(script_path, file_path)
    file_path = os.path.realpath(file_path)

    data = list(csv.reader(open(file_path), delimiter='\t'))
    for row in data:
        ratings[(int(row[0])) - 1][(int(row[1])) - 1] = int(row[2])
        rating_exists[(int(row[0])) - 1][(int(row[1])) - 1] = 1


def read_files(ratings, rating_exists):
    files = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base']
    for file in files:
        read(file, ratings, rating_exists)


def read_mvoies_meta_data(file):
    lines = open(os.path.join(os.path.dirname(__file__), file)).readlines()
    movie_meta = dict()
    for _ in lines:
        entires = _.split("|")
        movie_meta[entires[0]] = entires[1]
    return movie_meta


def read_input_data():
    """ Reads the input data from the 100k movielens dataset"""
    user_cnt = 943
    movie_cnt = 1682
    ratings = np.zeros([user_cnt, movie_cnt])
    rating_exists = np.zeros([user_cnt, movie_cnt])
    read_files(ratings, rating_exists)

    input_data = InputData()
    input_data.rating_mat = ratings
    input_data.is_rated_mat = rating_exists
    input_data.movie_cnt = movie_cnt
    input_data.user_cnt = user_cnt
    input_data.feature_cnt = 100
    return input_data


def main():
    read_input_data()


if __name__ == "__main__":
    main()
