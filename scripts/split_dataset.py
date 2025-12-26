import os
import random

def split_dataset(metadata_path, audio_dir, output_dir, val_count=200, ood_count=500):
    """
    Splits the metadata into train, validation, and OOD sets based on available audio files.
    """
    
    print(f"Reading metadata from: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines in metadata: {len(lines)}")

    # Filter metadata based on available audio files
    available_lines = []
    missing_count = 0
    
    print(f"Scanning for audio files in: {audio_dir}")
    # Get set of available files for faster lookup
    available_files = set(os.listdir(audio_dir))

    for line in lines:
        parts = line.strip().split('|')
        if not parts:
            continue
        
        filename = parts[0]
        # Check if file exists (assuming metadata filename matches audio filename)
        # Note: Metadata might or might not have .wav extension. based on previous view it has .wav
        if filename in available_files:
             available_lines.append(line)
        else:
            missing_count += 1

    print(f"Found {len(available_lines)} available audio files.")
    print(f"Missing {missing_count} files referenced in metadata.")

    if len(available_lines) < (val_count + ood_count):
        raise ValueError(f"Not enough data to split! Found {len(available_lines)}, but need at least {val_count + ood_count} for Val+OOD.")

    # Randomly shuffle the data
    print("Shuffling data...")
    random.seed(42) # For reproducibility
    random.shuffle(available_lines)

    # Split the data
    ood_data = available_lines[:ood_count]
    val_data = available_lines[ood_count : ood_count + val_count]
    train_data = available_lines[ood_count + val_count:]

    print(f"Split sizes:")
    print(f"  OOD: {len(ood_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Train: {len(train_data)}")

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # Write Train
    train_path = os.path.join(output_dir, 'train_list.txt')
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    print(f"Written train list to: {train_path}")

    # Write Val
    val_path = os.path.join(output_dir, 'val_list.txt')
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    print(f"Written val list to: {val_path}")

    # Write OOD (Text only)
    ood_path = os.path.join(output_dir, 'OOD_texts.txt')
    with open(ood_path, 'w', encoding='utf-8') as f:
        for line in ood_data:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                # Based on previous inspection, text is likely the 2nd element (index 1)
                # metadata format: filename|text|speaker
                phoneme_text = parts[1]
                f.write(phoneme_text + '\n')
    print(f"Written OOD texts to: {ood_path}")

if __name__ == "__main__":
    metadata_file = 'Data/metadata-styletts2-zir-phoneme-final.txt'
    audio_directory = 'Data/resampled'
    output_directory = 'Data'
    
    # Run the split
    split_dataset(metadata_file, audio_directory, output_directory)
