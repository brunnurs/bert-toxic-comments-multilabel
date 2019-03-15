from src import write_first_n_lines_of_file

if __name__ == "__main__":
    write_first_n_lines_of_file("/home/ursin/development/toxic-comments-with-bert/data/toxic_comments/train.large.csv",
                                "/home/ursin/development/toxic-comments-with-bert/data/toxic_comments/train.csv",
                                10000)

    write_first_n_lines_of_file("/home/ursin/development/toxic-comments-with-bert/data/toxic_comments/val.large.csv",
                                "/home/ursin/development/toxic-comments-with-bert/data/toxic_comments/val.csv",
                                10000)
