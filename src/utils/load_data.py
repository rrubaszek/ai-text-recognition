import pandas as pd
import csv
from utils.paths import DATASET_DIR

def load_data(filename: str = "data.tsv", type: str = "train", is_labels: bool = False):
    file_path = DATASET_DIR / type / filename

    try:
        # quoting=csv.QUOTE_NONE całkowicie wyłącza analizę cudzysłowów.
        # Dzięki temu żaden wiersz nie zostanie "zjedzony" przez niezamknięty cudzysłów.
        df = pd.read_csv(file_path, sep="\t", header=None, quoting=csv.QUOTE_NONE, low_memory=False)
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD ODCZYTU: {e}")
        return []

    if is_labels:
        col = df.iloc[:, -1]

        # Jeśli pierwszy wiersz to nagłówek, usuwamy go (korzystamy z .iloc dla bezpieczeństwa)
        if isinstance(col.iloc[0], str) and not str(col.iloc[0]).replace('.', '', 1).isdigit():
            col = col.iloc[1:]

        return col.astype(int).tolist()
    else:
        # Szukamy kolumny z tekstem
        lens = df.astype(str).map(len).mean()
        text_col_index = lens.idxmax()

        col = df[text_col_index]

        # Usuwamy nagłówek
        if str(col.iloc[0]).lower() in ['text', 'content', 'sentence', 'tweet', 'id']:
            col = col.iloc[1:]

        return col.astype(str).tolist()