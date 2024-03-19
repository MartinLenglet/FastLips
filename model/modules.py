import os
import json
import copy
import math
from collections import OrderedDict
from regex import B
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat

from utils.tools import get_mask_from_lengths, pad

from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        # Audio Variance Adaptor Config
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"][
            "normalization"
        ]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"][
            "normalization"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            # self.pitch_mean, self.pitch_std = stats["pitch"][2:4]
            energy_min, energy_max = stats["energy"][:2]
            # self.energy_mean, self.energy_std = stats["energy"][2:4]

            # Load Visual params
            lips_aperture_min, lips_aperture_max = stats["lips_aperture"][:2]
            # self.lips_aperture_mean, self.lips_aperture_std = stats["lips_aperture"][2:4]
            lips_spreading_min, lips_spreading_max = stats["lips_spreading"][:2]
            # self.lips_spreading_mean, self.lips_spreading_std = stats["lips_spreading"][2:4]

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats_by_speaker.json")
        ) as f:
            self.stats_by_speaker = json.load(f)

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats_lips_by_speaker.json")
        ) as f:
            self.stats_lips_by_speaker = json.load(f)

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        self.use_variance_predictor = model_config["use_variance_predictor"]
        self.use_variance_embeddings = model_config["use_variance_embeddings"]

        # Visual Variance Adaptor Config
        self.lips_aperture_feature_level = preprocess_config["preprocessing"]["lips_aperture"][
            "feature"
        ]
        self.lips_spreading_feature_level = preprocess_config["preprocessing"]["lips_spreading"][
            "feature"
        ]
        self.lips_aperture_normalization = preprocess_config["preprocessing"]["lips_aperture"][
            "normalization"
        ]
        self.lips_spreading_normalization = preprocess_config["preprocessing"]["lips_spreading"][
            "normalization"
        ]
        assert self.lips_aperture_feature_level in ["phoneme_level", "frame_level"]
        assert self.lips_spreading_feature_level in ["phoneme_level", "frame_level"]

        lips_aperture_quantization = model_config["variance_embedding_visual"]["lips_aperture_quantization"]
        lips_spreading_quantization = model_config["variance_embedding_visual"]["lips_spreading_quantization"]
        n_bins_visual = model_config["variance_embedding_visual"]["n_bins"]
        assert lips_aperture_quantization in ["linear", "log"]
        assert lips_spreading_quantization in ["linear", "log"]

        self.use_variance_predictor_visual = model_config["use_variance_predictor_visual"]
        self.use_variance_embeddings_visual = model_config["use_variance_embeddings_visual"]

        if self.use_variance_predictor_visual["lips_aperture"]:
            self.lips_aperture_predictor = VariancePredictor(model_config)

            if lips_aperture_quantization == "log":
                self.lips_aperture_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(lips_aperture_min), np.log(lips_aperture_max), n_bins_visual - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.lips_aperture_bins = nn.Parameter(
                    torch.linspace(lips_aperture_min, lips_aperture_max, n_bins_visual - 1),
                    requires_grad=False,
                )

            self.lips_aperture_embedding = nn.Embedding(
                n_bins_visual, model_config["transformer"]["encoder_hidden"]
            )

        if self.use_variance_predictor_visual["lips_spreading"]:    
            self.lips_spreading_predictor = VariancePredictor(model_config)

            if lips_spreading_quantization == "log":
                self.lips_spreading_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(lips_spreading_min), np.log(lips_spreading_max), n_bins_visual - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.lips_spreading_bins = nn.Parameter(
                    torch.linspace(lips_spreading_min, lips_spreading_max, n_bins_visual - 1),
                    requires_grad=False,
                )

            self.lips_spreading_embedding = nn.Embedding(
                n_bins_visual, model_config["transformer"]["encoder_hidden"]
            )

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.detach_energy_prediction = model_config["variance_predictor"]["detach_energy_prediction"]
        self.audio_to_visual_sampling_rate = preprocess_config["preprocessing"]["au"]["sampling_rate"]/(preprocess_config["preprocessing"]["audio"]["sampling_rate"]/preprocess_config["preprocessing"]["stft"]["hop_length"])

    def get_pitch_embedding(self, x, target, mask, control, speakers):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            # Pitch Prediction in Semitons => additive control
            if self.pitch_normalization:
                # self.pitch_std gives mean std from all speakers
                # prediction = prediction + control/self.pitch_std
                # prediction = prediction + control/3.5701

                # z-scores are normalized by speakers
                pitch_std_by_speaker = self.stats_by_speaker[list(self.stats_by_speaker.keys())[speakers]]["pitch"][3]
                prediction = prediction + control/pitch_std_by_speaker
            else:
                prediction = prediction + control

            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )

        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, speakers):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            # Energy Prediction in dB => additive control
            if self.energy_normalization:
                # self.energy_std gives mean std from all speakers
                # prediction = prediction + control/self.energy_std
                # prediction = prediction + control/8.4094

                # prediction = prediction + control/8.4094
                energy_std_by_speaker = self.stats_by_speaker[list(self.stats_by_speaker.keys())[speakers]]["energy"][3]
                prediction = prediction + control/energy_std_by_speaker
            else:
                prediction = prediction + control

            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    
    def get_lips_aperture_embedding(self, x, target, mask, control, speakers):
        prediction = self.lips_aperture_predictor(x, mask)
        if target is not None:
            embedding = self.lips_aperture_embedding(torch.bucketize(target, self.lips_aperture_bins))
        else:
            if self.lips_aperture_normalization:
                # z-scores are normalized by speakers
                try:
                    lips_aperture_std_by_speaker = self.stats_lips_by_speaker[list(self.stats_by_speaker.keys())[speakers]]["lips_aperture"][3]
                except:
                    lips_aperture_std_by_speaker = self.stats_lips_by_speaker["AD"]["lips_aperture"][3]

                prediction = prediction + control/lips_aperture_std_by_speaker
            else:
                prediction = prediction + control

            embedding = self.lips_aperture_embedding(
                torch.bucketize(prediction, self.lips_aperture_bins)
            )

        return prediction, embedding
    
    def get_lips_spreading_embedding(self, x, target, mask, control, speakers):
        prediction = self.lips_spreading_predictor(x, mask)
        if target is not None:
            embedding = self.lips_spreading_embedding(torch.bucketize(target, self.lips_spreading_bins))
        else:
            if self.lips_spreading_normalization:
                # z-scores are normalized by speakers
                try: 
                    lips_spreading_std_by_speaker = self.stats_lips_by_speaker[list(self.stats_by_speaker.keys())[speakers]]["lips_spreading"][3]
                except:
                    lips_spreading_std_by_speaker = self.stats_lips_by_speaker["AD"]["lips_spreading"][3]

                prediction = prediction + control/lips_spreading_std_by_speaker
            else:
                prediction = prediction + control

            embedding = self.lips_spreading_embedding(
                torch.bucketize(prediction, self.lips_spreading_bins)
            )

        return prediction, embedding
    
    def compensate_rounding_duration(self, raw_duration):
        predicted_duration_compensated = raw_duration.clone()

        for utt_in_batch in range(raw_duration.size()[0]):
            residual = 0.0
            for index_phon in range(raw_duration.size()[1]):
                dur_phon = raw_duration[utt_in_batch][index_phon]
                dur_phon_rounded = torch.round(dur_phon + residual)
                residual += dur_phon - dur_phon_rounded
                predicted_duration_compensated[utt_in_batch][index_phon] = dur_phon_rounded

        # Add residual to compensate for round
        duration_rounded = torch.clamp(
            predicted_duration_compensated,
            min=0,
        )
        # may be updated with torch.cumsum?
        
        return duration_rounded

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_mel_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=0.0,
        e_control=0.0,
        d_control=1.0,
        src_mask_noSpectro=None,
        speakers=None,
        au_mask=None,
        max_au_len=None,
        lips_aperture_target=None,
        lips_spreading_target=None,
        la_control=0.0,
        ls_control=0.0,
    ):

        # log_duration_prediction = self.duration_predictor(x, src_mask)
        log_duration_prediction = self.duration_predictor(x, src_mask_noSpectro)

        x_au = x.clone() # for visual prediction, "clone" duplicates the tensor and saves the gradient

        # ---------- Compute explicit predictors at phoneme-level ----------------
        if not self.training or mel_mask.nelement():
            # AUDIO: Pitch
            if self.pitch_feature_level == "phoneme_level":
                if self.use_variance_predictor["pitch"]:
                    # pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    #     x, pitch_target, src_mask, p_control
                    # )
                    pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                        x, pitch_target, src_mask_noSpectro, p_control, speakers
                    )

                    if self.use_variance_embeddings["pitch"] and not self.detach_energy_prediction:
                        x = x + pitch_embedding
                else:
                    pitch_prediction = None
                    
            # AUDIO: Energy
            if self.energy_feature_level == "phoneme_level":
                if self.use_variance_predictor["energy"]:
                    # energy_prediction, energy_embedding = self.get_energy_embedding(
                    #     x, energy_target, src_mask, e_control
                    # )
                    energy_prediction, energy_embedding = self.get_energy_embedding(
                        x, energy_target, src_mask_noSpectro, e_control, speakers
                    )

                    if self.use_variance_embeddings["energy"] and not self.detach_energy_prediction:
                        x = x + energy_embedding
                else:
                    energy_prediction = None

            # Handle Cascaded Prediction
            if self.detach_energy_prediction:
                if self.pitch_feature_level == "phoneme_level" and self.use_variance_embeddings["pitch"]:
                    x = x + pitch_embedding

                if self.energy_feature_level == "phoneme_level" and self.use_variance_embeddings["energy"]:
                    x = x + energy_embedding
        else:
            if self.pitch_feature_level == "phoneme_level":
                pitch_prediction = None
            if self.energy_feature_level == "phoneme_level":
                energy_prediction = None

        # VISUAL: Lips Aperture
        if not self.training or au_mask.nelement():
            if self.lips_aperture_feature_level == "phoneme_level":
                if self.use_variance_predictor_visual["lips_aperture"]:
                    lips_aperture_prediction, lips_aperture_embedding = self.get_lips_aperture_embedding(
                        x_au, lips_aperture_target, src_mask_noSpectro, la_control, speakers
                    )
                else:
                    lips_aperture_prediction = None

            # VISUAL: Lips Spreading
            if self.lips_spreading_feature_level == "phoneme_level":
                if self.use_variance_predictor_visual["lips_spreading"]:
                    lips_spreading_prediction, lips_spreading_embedding = self.get_lips_spreading_embedding(
                        x_au, lips_spreading_target, src_mask_noSpectro, ls_control, speakers
                    )
                else:
                    lips_spreading_prediction = None

            # Add both embeddings after prediction
            if self.lips_aperture_feature_level == "phoneme_level" and self.use_variance_embeddings_visual["lips_aperture"]:
                x_au = x_au + lips_aperture_embedding

            if self.lips_spreading_feature_level == "phoneme_level" and self.use_variance_embeddings_visual["lips_spreading"]:
                x_au = x_au + lips_spreading_embedding
        else:
            if self.lips_aperture_feature_level == "phoneme_level":
                lips_aperture_prediction = None
            if self.lips_spreading_feature_level == "phoneme_level":
                lips_spreading_prediction = None

        # ---------- Length Regulator ----------------
        if duration_target is not None:
            duration_target_au = self.compensate_rounding_duration(torch.mul(duration_target, self.audio_to_visual_sampling_rate))

            x, mel_len = self.length_regulator(x, duration_target, max_mel_len)
            x_au, au_len = self.length_regulator(x_au, duration_target_au, max_au_len)

            duration_rounded = duration_target
        else:
            predicted_duration = (torch.exp(log_duration_prediction) - 1) * d_control
            duration_rounded = self.compensate_rounding_duration(predicted_duration)
            duration_rounded_au = self.compensate_rounding_duration(torch.mul(duration_rounded, self.audio_to_visual_sampling_rate))

            x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)
            mel_mask = get_mask_from_lengths(mel_len)

            x_au, au_len = self.length_regulator(x_au, duration_rounded_au, max_au_len)
            au_mask = get_mask_from_lengths(au_len)

        # ---------- Compute explicit predictors at frame-level ----------------
        if not self.training or mel_mask.nelement():
            # AUDIO: Pitch
            if self.pitch_feature_level == "frame_level":
                if self.use_variance_predictor["pitch"]:
                    pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                        x, pitch_target, mel_mask, p_control, speakers
                    )
                else:
                    pitch_prediction = None

            # AUDIO: Energy
            if self.energy_feature_level == "frame_level":
                if self.use_variance_predictor["energy"]:
                    energy_prediction, energy_embedding = self.get_energy_embedding(
                        x, energy_target, mel_mask, e_control, speakers
                    )
                else:
                    energy_prediction = None

            # AUDIO: Add frame-level embeddings
            if self.pitch_feature_level == "frame_level" and self.use_variance_embeddings["pitch"]:
                x = x + pitch_embedding

            if self.energy_feature_level == "frame_level" and self.use_variance_embeddings["energy"]:
                x = x + energy_embedding
        else:
            if self.pitch_feature_level == "frame_level":
                pitch_prediction = None
            if self.energy_feature_level == "frame_level":
                energy_prediction = None

        # VISUAL: lips_aperture
        if not self.training or au_mask.nelement():
            if self.lips_aperture_feature_level == "frame_level":
                if self.use_variance_predictor_visual["lips_aperture"]:
                    lips_aperture_prediction, lips_aperture_embedding = self.get_lips_aperture_embedding(
                        x_au, lips_aperture_target, au_mask, la_control, speakers
                    )
                else:
                    lips_aperture_prediction = None
                
            # VISUAL: lips_spreading
            if self.lips_spreading_feature_level == "frame_level":
                if self.use_variance_predictor_visual["lips_spreading"]:
                    lips_spreading_prediction, lips_spreading_embedding = self.get_lips_spreading_embedding(
                        x_au, lips_spreading_target, au_mask, ls_control, speakers
                    )
                else:
                    lips_spreading_prediction = None

            # VISUAL: Add frame-level embeddings
            if self.lips_aperture_feature_level == "frame_level" and self.use_variance_embeddings_visual["lips_aperture"]:
                x_au = x_au + lips_aperture_embedding

            if self.lips_spreading_feature_level == "frame_level" and self.use_variance_embeddings_visual["lips_spreading"]:
                x_au = x_au + lips_spreading_embedding
        else:
            if self.lips_aperture_feature_level == "frame_level":
                lips_aperture_prediction = None
            if self.lips_spreading_feature_level == "frame_level":
                lips_spreading_prediction = None

        return (
            x,
            x_au,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
            lips_aperture_prediction,
            lips_spreading_prediction,
            au_len,
            au_mask,
        )

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
            mel_len = [min(single_mel_len, max_len) for single_mel_len in mel_len] # Remove last frame in case of rounding error
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()

            # out.append(vec.expand(max(int(expand_size), 0), -1))
            out.append(vec.expand(max(int(np.round(expand_size)), 0), -1))

        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            # padding_mode='replicate',
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        # return F.softmax(self.linear_layer(x), dim=2)
        return self.linear_layer(x) # CrossEntropyLoss computes softmax internally
