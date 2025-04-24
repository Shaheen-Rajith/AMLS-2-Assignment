import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def generate_spectrograms(input_dir, output_dir, sample_rate=32000, duration=5, n_mels=128):
    # make output folder if not exists already
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing batch in {root}"):
            if not file.endswith('.ogg'):
                continue
            # mimicing the same file structure for the output spectrograms as well
            input_path = os.path.join(root, file)
            label = os.path.basename(root)
            output_label_dir = os.path.join(output_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)
            output_path = os.path.join(output_label_dir, file.replace('.ogg', '.png'))

            try:
                y, sr = librosa.load(input_path, sr=sample_rate, duration=duration)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                plt.figure(figsize=(2.24, 2.24), dpi=100) # 224 x 224
                plt.axis('off')
                librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"Error processing {input_path}: {e}")
