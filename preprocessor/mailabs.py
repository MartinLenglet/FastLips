import os

import librosa
import json
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from text import _clean_text

import re 

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    csv_dir = config["path"]["csv_path"]
    out_dir = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    au_dir = config["path"]["au_path"]
    au_config = config["preprocessing"]["au"]
    os.makedirs((os.path.join(preprocessed_dir, "au")), exist_ok=True)

    as_dir = config["path"]["as_path"]
    os.makedirs((os.path.join(preprocessed_dir, "lips_aperture")), exist_ok=True)
    os.makedirs((os.path.join(preprocessed_dir, "lips_spreading")), exist_ok=True)

    csv_name = "NEB_train.csv"
    base_name_pre = ''
    upsample_au = True
    normalize_lips_features = True

    with open(os.path.join(csv_dir, csv_name), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            start_utt = parts[1]
            start_utt = int(int(start_utt) * sampling_rate / 1000)
            end_utt = parts[2]
            end_utt = int(int(end_utt) * sampling_rate / 1000)
            text = parts[3]
            text = _clean_text(text, cleaners)
            align = parts[4]

            if re.match('.*_NEB_.*', base_name):

                print(base_name)

                if len(base_name.split("_")) >= 6:
                        speaker = base_name.split("_")[3]
                else:
                        speaker = base_name.split("_")[2]

                wav_path = os.path.join(in_dir, "{}.wav".format(base_name))

                number_utt_in_chapter = 1

                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

                    if base_name_pre != base_name:
                        wav_chapter, _ = librosa.load(wav_path, sr=sampling_rate)
                        base_name_pre = base_name
                        number_utt_in_chapter = 1
                    else:
                        number_utt_in_chapter += 1

                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav[start_utt:end_utt]
                    wav = wav / max(abs(wav)) * max_wav_value * 0.95

                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}_{}.wav".format(base_name, number_utt_in_chapter)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )

                    # Write Action Units (AU) as numpy array when exists
                    au_path = os.path.join(au_dir, "{}.AU".format(base_name))
                    if os.path.exists(au_path) and True:
                        start_utt = parts[1]
                        start_utt_au = int(round(int(start_utt) * au_config["sampling_rate"] / 1000))
                        end_utt = parts[2]
                        end_utt_au = int(round(int(end_utt) * au_config["sampling_rate"] / 1000))

                        (lg_data_visual, dim_visual, num, den) = tuple(np.fromfile(au_path, count=4, dtype=np.int32))
                        lg_visual = end_utt_au - start_utt_au
                        visual_params = np.memmap(au_path, offset=16+(start_utt_au * dim_visual * 4), dtype=np.float32, shape=(lg_visual, dim_visual)).transpose()

                        # perform linear interpolation to match mel frame sampling rate
                        if upsample_au:
                            factor_interp = (sampling_rate/hop_length)/au_config["sampling_rate"]
                            # print(factor_interp)
                            size_interp = round(visual_params[0, :].size*factor_interp)
                            visual_params_interp = np.zeros((visual_params[:, 0].size, size_interp))
                            for i in range(0, visual_params[:,0].size):
                                visual_params_interp[i] = np.interp(np.linspace(0, 1, size_interp), np.linspace(0, 1, visual_params[0, :].size), visual_params[i, :])
                            # print(visual_params_interp[0,:].size)

                            au_filename = "{}-au-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter)
                            np.save(
                                os.path.join(preprocessed_dir, "au", au_filename),
                                visual_params_interp.transpose(),
                            )
                        else:
                            au_filename = "{}-au-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter)
                            np.save(
                                os.path.join(preprocessed_dir, "au", au_filename),
                                visual_params.transpose(),
                            )

                        as_path = os.path.join(as_dir, "{}.AB".format(base_name))
                        if os.path.exists(as_path) and True:
                            (lg_data_ab, dim_ab, num_ab, den_ab) = tuple(np.fromfile(as_path, count=4, dtype=np.int32))
                            as_params = np.memmap(as_path, offset=16+(start_utt_au * dim_ab * 4), dtype=np.float32, shape=(lg_visual, dim_ab)).transpose()
                            
                            lips_aperture_filename = "{}-lips_aperture-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter)
                            lips_spreading_filename = "{}-lips_spreading-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter)

                            np.save(
                                os.path.join(preprocessed_dir, "lips_aperture", lips_aperture_filename),
                                as_params[0, :],
                            )
                            np.save(
                                os.path.join(preprocessed_dir, "lips_spreading", lips_spreading_filename),
                                as_params[1, :],
                            )

    # Normalize lips features
    if normalize_lips_features:
        lips_aperture_scaler_all_speakers = StandardScaler()
        lips_spreading_scaler_all_speakers = StandardScaler()
        lips_aperture_max_all_speakers = np.finfo(np.float64).min
        lips_aperture_min_all_speakers = np.finfo(np.float64).max
        lips_spreading_max_all_speakers = np.finfo(np.float64).min
        lips_spreading_min_all_speakers = np.finfo(np.float64).max

        speakers = [
            "AD",
        ]
        stats_by_speaker = {}

        for speaker in speakers:
            lips_aperture_scaler = StandardScaler()
            lips_spreading_scaler = StandardScaler()

            base_name_pre = ''

            with open(os.path.join(csv_dir, csv_name), encoding="utf-8") as f:
                for line in tqdm(f):
                    parts = line.strip().split("|")
                    base_name = parts[0]
                    start_utt = parts[1]
                    start_utt = int(round(int(start_utt) * sampling_rate / 1000))
                    end_utt = parts[2]
                    end_utt = int(round(int(end_utt) * sampling_rate / 1000))
                    text = parts[3]
                    text = _clean_text(text, cleaners)
                    align = parts[4]

                    if base_name_pre != base_name:
                        number_utt_in_chapter = 1
                        base_name_pre = base_name
                    else:
                        number_utt_in_chapter += 1

                    if re.match(".*_{}_.*".format(speaker), base_name):

                        print(base_name)

                        lips_aperture_path = os.path.join(
                            preprocessed_dir,
                            "lips_aperture",
                            "{}-lips_aperture-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter),
                        )
                        if os.path.exists(lips_aperture_path):
                            lips_aperture = np.load(lips_aperture_path)

                            if len(lips_aperture) > 0:
                                lips_aperture_scaler.partial_fit(lips_aperture.reshape((-1, 1)))
                                lips_aperture_scaler_all_speakers.partial_fit(lips_aperture.reshape((-1, 1)))

                        lips_spreading_path = os.path.join(
                            preprocessed_dir,
                            "lips_spreading",
                            "{}-lips_spreading-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter),
                        )
                        if os.path.exists(lips_spreading_path):
                            lips_spreading = np.load(lips_spreading_path)
                            
                            if len(lips_spreading) > 0:
                                lips_spreading_scaler.partial_fit(lips_spreading.reshape((-1, 1)))
                                lips_spreading_scaler_all_speakers.partial_fit(lips_spreading.reshape((-1, 1)))

            print("Computing statistic quantities for speaker:{} ...".format(speaker))
            lips_aperture_mean = lips_aperture_scaler.mean_[0]
            lips_aperture_std = lips_aperture_scaler.scale_[0]
            lips_spreading_mean = lips_spreading_scaler.mean_[0]
            lips_spreading_std = lips_spreading_scaler.scale_[0]

            # Normalization by speaker
            lips_aperture_min, lips_aperture_max = normalize(
                os.path.join(preprocessed_dir, "lips_aperture"), lips_aperture_mean, lips_aperture_std, speaker
            )
            lips_spreading_min, lips_spreading_max = normalize(
                os.path.join(preprocessed_dir, "lips_spreading"), lips_spreading_mean, lips_spreading_std, speaker
            )

            stats_by_speaker[speaker] = {
                "lips_aperture": [
                    float(lips_aperture_min),
                    float(lips_aperture_max),
                    float(lips_aperture_mean),
                    float(lips_aperture_std),
                ],
                "lips_spreading": [
                    float(lips_spreading_min),
                    float(lips_spreading_max),
                    float(lips_spreading_mean),
                    float(lips_spreading_std),
                ],
            }

            with open(os.path.join(preprocessed_dir, "stats_lips_{}.json".format(speaker)), "w") as f:
                f.write(json.dumps(stats_by_speaker[speaker]))

            lips_aperture_min_all_speakers = min(lips_aperture_min_all_speakers, lips_aperture_min)
            lips_aperture_max_all_speakers = max(lips_aperture_max_all_speakers, lips_aperture_max)
            lips_spreading_min_all_speakers = min(lips_spreading_min_all_speakers, lips_spreading_min)
            lips_spreading_max_all_speakers = max(lips_spreading_max_all_speakers, lips_spreading_max)

        # Global lips stats
        lips_aperture_mean_all_speakers = lips_aperture_scaler_all_speakers.mean_[0]
        lips_aperture_std_all_speakers = lips_aperture_scaler_all_speakers.scale_[0]
        lips_spreading_mean_all_speakers = lips_spreading_scaler_all_speakers.mean_[0]
        lips_spreading_std_all_speakers = lips_spreading_scaler_all_speakers.scale_[0]

        with open(os.path.join(preprocessed_dir, "stats_lips_by_speaker.json"), "w") as f:
            f.write(json.dumps(stats_by_speaker))

        # Save files
        with open(os.path.join(preprocessed_dir, "speakers_lips.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(preprocessed_dir, "stats_lips.json"), "w") as f:
            stats = {
                "lips_aperture": [
                    float(lips_aperture_min_all_speakers),
                    float(lips_aperture_max_all_speakers),
                    float(lips_aperture_mean_all_speakers),
                    float(lips_aperture_std_all_speakers),
                ],
                "lips_spreading": [
                    float(lips_spreading_min_all_speakers),
                    float(lips_spreading_max_all_speakers),
                    float(lips_spreading_mean_all_speakers),
                    float(lips_spreading_std_all_speakers),
                ],
            }
            f.write(json.dumps(stats))

def normalize(in_dir, mean, std, speaker):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        if re.match(".*_{}_.*".format(speaker), filename):
            filename = os.path.join(in_dir, filename)
            print(filename)

            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

    return min_value, max_value