import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)

        # model.load_state_dict(ckpt["model"]) # load models and rises error if missing key

        # Load Pre-trained model (ignore missing keys and mismatch sizes)
        pretrained_dict = ckpt["model"]
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and (model_dict[k].shape == pretrained_dict[k].shape))}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            # scheduled_optim.load_state_dict(ckpt["optimizer"])
            try:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
            except:
                print("Optimizer could not be loaded from checkpoint. Start optimizer from scratch.")
        model.train()

        # Case Freeze Encoder
        if args.freeze_encoder:
            model.freeze_encoder()
        # Case Freeze Decoder
        if args.freeze_decoder:
            model.freeze_decoder()
        # Case Freeze Visual Decoder
        if args.freeze_decoder_visual:
            model.freeze_decoder_visual()
        # Case Freeze Postnet
        if args.freeze_postnet:
            model.freeze_postnet()
        # Case Freeze Postnet
        if args.freeze_postnet_visual:
            model.freeze_postnet_visual()
        # Case Speaker Embeddings
        if args.freeze_speaker_emb:
            model.freeze_speaker_emb()
        # Case Phon Prediction
        if args.freeze_phon_prediction:
            model.freeze_phon_prediction()
        # Case Variance Prediction
        if args.freeze_variance_prediction:
            model.freeze_variance_prediction()

        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
