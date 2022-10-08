import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display



def create_spectrogram(y):
    spec = librosa.feature.melspectrogram(y=y)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    return spec_conv


def save_spectrogram(spectrogram, file_name, classid):
    if str(classid) not in os.listdir("spectrograms"):
        os.mkdir(f"spectrograms/{classid}")

    save_name = file_name.split(".")[0]
    
    plt.figure()
    librosa.display.specshow(spectrogram)
    plt.savefig(f"spectrograms/{classid}/{save_name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

df = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

path_to_folds = "UrbanSound8K/audio"

if "spectrograms" not in os.listdir():
    os.mkdir("spectrograms")


number_of_files = df.shape[0]
number_of_processed = 0
number_of_errors = 0


with open("errors.txt", "w") as error_file:
    for index, row in df.iterrows():
        try:
            file_name = row["slice_file_name"]
            fold = row["fold"]
            classid = row["classID"]
            path_to_file = f"{path_to_folds}/fold{fold}/{file_name}"
                        
            data, sr = librosa.load(path_to_file)
            spectrogram = create_spectrogram(data)
            save_spectrogram(spectrogram, file_name, classid)
            
            del data
            del sr
            del spectrogram
            
        except Exception as e:
            number_of_errors += 1
            error_file.write(f"{number_of_errors}: {e}\n")
        
        finally:
            number_of_processed += 1
        
        print(f"\rNumber: {number_of_processed}/{number_of_files} | Errors: {number_of_errors}", end="")