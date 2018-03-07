import csv
import os


def save(result, filename, path_up=''):
    path = os.path.abspath(os.path.join(path_up, 'results', filename))
    with open(path, 'w', newline='') as my_csv:
        csv_writer = csv.writer(my_csv)
        csv_writer.writerows(result)
