from preprocessing.run import run as run_preprocessing
from models.run import run as run_models

def main():
    print("Starting preprocessing...")
    run_preprocessing(force=False)
    
    print("\nStarting model training and evaluation...")
    # Always use lodo=False until we add domains to the dataset end implement LODO in preprocessing
    run_models(run_cv=False, optimize=True, lodo=False, shap=False)
    
if __name__ == '__main__':
    main()