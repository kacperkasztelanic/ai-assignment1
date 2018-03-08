import csv
import os


class DataSaver:
    def __init__(self, results_dir, results_ext):
        self.results_dir = results_dir
        self.results_ext = results_ext

    def save(self, result, filename) -> None:
        path = os.path.join(self.results_dir, (filename + '.' + self.results_ext))
        with open(path, 'w', newline='') as my_csv:
            csv_writer = csv.writer(my_csv)
            csv_writer.writerows(result)
