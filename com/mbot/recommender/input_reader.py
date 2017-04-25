import numpy as np
import csv


class InputData:
    user_cnt = None
    movie_cnt = None
    feature_cnt = None
    rating_mat = None
    is_rated_mat = None


def read(file, ratings):
    file_path = ''
    data = list(csv.reader(open(file_path), delimiter='\t'))
    for row in data:
        ratings[int(row[0])][int(row[1])] = int(row[2])


def read_files(ratings):
    files = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base']
    for file in files:
        read(file, ratings)


def read_input_data():
    input = InputData()
    return input


def main():
    user_cnt = 944
    movie_cnt = 1683
    ratings = np.zeros([user_cnt, movie_cnt])
    read_files(ratings)


if __name__ == "__main__":
    main()
