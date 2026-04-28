import csv

from utils.paths import DATASET_DIR

def load_data(filename: str = "data.tsv", type: str = "train"):
    data = []
    with open(DATASET_DIR / type / filename) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if not row:
                continue
            
            data.append(row[0])
    
    return data
