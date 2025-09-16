import os

FOLDER_DATA      = "data"
FOLDER_RAW       = os.path.join(FOLDER_DATA, "raw")
FOLDER_PROCESSED = os.path.join(FOLDER_DATA, "processed")
FOLDER_COMBINED  = os.path.join(FOLDER_DATA, "combined")
FOLDER_DB        = os.path.join(FOLDER_DATA, "db")

SEED             = 123

def main():
    # create folders if they don't exist
    for folder in [FOLDER_DATA, FOLDER_RAW, FOLDER_PROCESSED, FOLDER_COMBINED, FOLDER_DB]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

if __name__ == "__main__":
    main()

