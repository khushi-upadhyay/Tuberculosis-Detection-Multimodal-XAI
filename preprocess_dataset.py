# preprocess_dataset.py

from utils.preprocess import save_preprocessed_data

if __name__ == "__main__":
    # Only Shenzhen
    save_preprocessed_data(
        root_dir='data/',
        output_path='processed_shenzhen.pt',
        use_montgomery=False
    )

    # # Optional: With Montgomery
    # save_preprocessed_data(
    #     root_dir='data/',
    #     output_path='processed_combined.pt',
    #     use_montgomery=True
    # )
