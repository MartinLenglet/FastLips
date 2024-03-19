import re
import argparse
from string import punctuation
import os
import json

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, synth_samples, expand, plot_mel
from dataset import TextDataset, Dataset
from text import text_to_sequence, out_symbols

from matplotlib import pyplot as plt

from scipy.io import savemat
from scipy.signal import butter,filtfilt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def process_per_batch(
    batch, 
    model, 
    pitch_control, 
    energy_control, 
    duration_control,
    lip_aperture_control,
    lip_spreading_control, 
    output_wav, 
    vocoder, 
    model_config, 
    preprocess_config, 
    train_config, 
    extension_data,
    mel_dim,
    no_spectro=False
):
    batch = to_device(batch, device)

    with torch.no_grad():
        # Forward
        output = model(
            *(batch[2:]),
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control,
            la_control=lip_aperture_control,
            ls_control=lip_spreading_control,
            no_spectro=no_spectro,
        )

        if output_wav:
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
        else:
            output_syn_path = preprocess_config["path"]["output_syn_path"]
            for i in range(len(output[8])):
                # Original Name
                basename = batch[0][i]
                src_len = output[8][i].item()

                # If no_spectro, compute only phon prediction
                if no_spectro:
                    if model_config["compute_phon_prediction"]:
                        phon_targets = batch[6][i, :src_len].transpose()
                        phon_prediction = output[10][i, :, :src_len].detach().cpu().numpy()

                        # save phon prediction
                        ind_pred = phon_prediction.argmax(axis=0)
                        phon_pred = []
                        for p in range(len(phon_targets)):
                            if phon_targets[p]==-1:
                                phon_pred.append("None|{}".format(out_symbols[ind_pred[p]]))
                            else:
                                phon_pred.append("{}|{}".format(out_symbols[phon_targets[p]], out_symbols[ind_pred[p]]))

                        mdic = {"phon_prediction_mat": phon_pred}
                        nm_phon_pred = '{}/{}_phon.mat'.format(output_syn_path, basename)
                        savemat(nm_phon_pred, mdic)
                        print('{}/{}_phon.mat created'.format(output_syn_path, basename, extension_data), flush=True)
                    continue

                mel_len = output[9][i].item()
                mel_prediction = output[1][i, :mel_len].detach().transpose(0, 1).cpu().data.numpy().transpose()

                # Copy of the spectrum in a file
                fp = open('{}/{}.{}'.format(output_syn_path, basename, extension_data), 'wb')
                fp.write(np.asarray((mel_len, mel_dim), dtype=np.int32))
                fp.write(mel_prediction.copy(order='C'))
                fp.close()
                print('{}/{}.{} created'.format(output_syn_path, basename, extension_data), flush=True)

                log_duration = output[4][i, :src_len].detach().cpu().numpy()
                duration = output[5][i, :src_len].detach().cpu().numpy()

                if model_config["save_predictions"]["duration"]:
                    # save duration in .mat format
                    mdic = {"duration_mat": duration}
                    nm_duration = '{}/{}_duration.mat'.format(output_syn_path, basename)
                    savemat(nm_duration, mdic)

                if model_config["save_predictions"]["log_duration"]:
                    # save log_duration in .mat format
                    mdic = {"log_duration_mat": log_duration}
                    nm_duration = '{}/{}_log_duration.mat'.format(output_syn_path, basename)
                    savemat(nm_duration, mdic)

                if model_config["use_variance_predictor"]["pitch"] and model_config["save_predictions"]["pitch"]:
                    # save pitch prediction in .mat format
                    pitch_prediction = output[2][i, :src_len].detach().cpu().numpy().transpose()

                    mdic = {"pitch_prediction_mat": pitch_prediction}
                    nm_pitch = '{}/{}_pitch.mat'.format(output_syn_path, basename)
                    savemat(nm_pitch, mdic)

                if model_config["use_variance_predictor"]["energy"] and model_config["save_predictions"]["energy"]:
                    # save energy prediction in .mat format
                    energy_prediction = output[3][i, :src_len].detach().cpu().numpy().transpose()

                    mdic = {"energy_prediction_mat": energy_prediction}
                    nm_energy = '{}/{}_energy.mat'.format(output_syn_path, basename)
                    savemat(nm_energy, mdic)

                if model_config["use_variance_predictor_visual"]["lips_aperture"] and model_config["save_predictions"]["lips_aperture"]:
                    # save lips_aperture prediction in .mat format
                    lips_aperture_prediction = output[16][i, :src_len].detach().cpu().numpy().transpose() # verif index 

                    mdic = {"lips_aperture_prediction_mat": lips_aperture_prediction}
                    nm_lips_aperture = '{}/{}_lips_aperture.mat'.format(output_syn_path, basename)
                    savemat(nm_lips_aperture, mdic)

                if model_config["use_variance_predictor_visual"]["lips_spreading"] and model_config["save_predictions"]["lips_spreading"]:
                    # save lips_spreading prediction in .mat format
                    lips_spreading_prediction = output[17][i, :src_len].detach().cpu().numpy().transpose() # verif index 

                    mdic = {"lips_spreading_prediction_mat": lips_spreading_prediction}
                    nm_lips_spreading = '{}/{}_lips_spreading.mat'.format(output_syn_path, basename)
                    savemat(nm_lips_spreading, mdic)

                if model_config["compute_phon_prediction"] and model_config["save_predictions"]["phon"]:
                    phon_targets = batch[6][i, :src_len].transpose()
                    phon_prediction = output[10][i, :, :src_len].detach().cpu().numpy()
                    ind_pred = phon_prediction.argmax(axis=0)

                    phon_pred = []
                    for p in range(len(phon_targets)):
                        if phon_targets[p]==-1:
                            phon_pred.append("None|{}".format(out_symbols[ind_pred[p]]))
                        else:
                            # print("{}|{}".format(out_symbols[phon_targets[p]], out_symbols[ind_pred[p]]))
                            phon_pred.append("{}|{}".format(out_symbols[phon_targets[p]], out_symbols[ind_pred[p]]))

                    mdic = {"phon_prediction_mat": phon_pred}
                    nm_phon_pred = '{}/{}_phon.mat'.format(output_syn_path, basename)
                    savemat(nm_phon_pred, mdic)

                if model_config["visual_prediction"]["compute_visual_prediction"] and model_config["save_predictions"]["au"]:                   
                    au_len = output[19][i].item() # verif index 
                    if model_config["visual_prediction"]["visual_postnet"]:
                        au_prediction = output[15][i, :au_len].detach().transpose(0, 1).cpu().data.numpy().transpose() # verif index 
                    else:
                        au_prediction = output[14][i, :au_len].detach().transpose(0, 1).cpu().data.numpy().transpose() # verif index 

                    # Copy of the action units in a file
                    extension_au = model_config["visual_prediction"]["extension"]
                    au_dim = preprocess_config["preprocessing"]["au"]["n_units"]
                    num = preprocess_config["preprocessing"]["au"]["sampling_rate"]
                    den = 1

                    # Optional denoising
                    if model_config["au_smoothing"]["use_smoothing"]:
                        for i_au in range(6): # 6 first parameters are for the head movements
                            au_prediction[:, i_au] = butter_lowpass_filter(au_prediction[:, i_au], model_config["au_smoothing"]["cutoff"], num/den, 1) # cutoff = model_config["au_smoothing"]["cutoff"] Hz / order = 1

                    fp = open('{}/{}.{}'.format(output_syn_path, basename, extension_au), 'wb')
                    fp.write(np.asarray((au_len, au_dim, num, den), dtype=np.int32))
                    fp.write(au_prediction.copy(order='C'))
                    fp.close()
                    print('{}/{}.{} created'.format(output_syn_path, basename, extension_au), flush=True)

                if False:
                    mel_prediction = output[1][i, :mel_len].detach().transpose(0, 1)
                    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
                        pitch = output[2][i, :src_len].detach().cpu().numpy()
                        pitch = expand(pitch, duration)
                    else:
                        pitch = output[2][i, :mel_len].detach().cpu().numpy()
                    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
                        energy = output[3][i, :src_len].detach().cpu().numpy()
                        energy = expand(energy, duration)
                    else:
                        energy = output[3][i, :mel_len].detach().cpu().numpy()

                    with open(
                        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
                    ) as f:
                        stats = json.load(f)
                        stats = stats["pitch"] + stats["energy"][:2]

                    fig = plot_mel(
                        [
                            (mel_prediction.cpu().numpy(), pitch, energy),
                        ],
                        stats,
                        ["Synthetized Spectrogram"],
                    )
                    plt.savefig('output/spectro/{}.png'.format(basename))
                    plt.close()

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def synthesize(model, step, configs, vocoder, batchs, control_values, no_spectro=False):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control, lip_aperture_control, lip_spreading_control = control_values

    # Output of the model
    output_wav = train_config["output"]["wav"]
    extension_data = model_config["vocoder"]["model"]
    mel_dim = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]

    for batch in batchs:
        process_per_batch(
            batch, 
            model, 
            pitch_control, 
            energy_control, 
            duration_control, 
            lip_aperture_control,
            lip_spreading_control,
            output_wav, 
            vocoder, 
            model_config, 
            preprocess_config, 
            train_config, 
            extension_data,
            mel_dim,
            no_spectro,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=0.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=0.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--no_spectro",
        required=False,
        action='store_true',
        help="Generation stops before variance adaptor and decoder",
    )
    parser.add_argument(
        "--lip_aperture_control",
        type=float,
        default=0.0,
        help="control the lip aperture of the whole utterance, larger value for more aperture",
    )
    parser.add_argument(
        "--lip_spreading_control",
        type=float,
        default=0.0,
        help="control the lip spreading of the whole utterance, larger value for more spreading",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        batch_size=1
        assert args.source is not None and args.text is None
    if args.mode == "single":
        batch_size=1
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    
    # Output of the model
    output_wav = train_config["output"]["wav"]

    # Get model
    model = get_model(args, configs, device, train=False)

    if output_wav:
        # Load vocoder
        vocoder = get_vocoder(model_config, device)
    else:
        vocoder = None

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
        )

    if args.mode == "single":
        cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        phones = np.array([text_to_sequence(raw_texts[0], cleaners)])
        text_lens = np.array([len(phones[0])])
        batchs = [(ids, raw_texts, speakers, phones, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control, args.lip_aperture_control, args.lip_spreading_control

    synthesize(
        model,
        args.restore_step, 
        configs, 
        vocoder, 
        batchs, 
        control_values, 
        no_spectro=args.no_spectro,
    )
