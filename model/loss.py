import torch
import torch.nn as nn
from utils.tools import device

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]

        self.use_variance_predictor = model_config["use_variance_predictor"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.compute_phon_prediction = model_config["compute_phon_prediction"]

        self.compute_visual_prediction = model_config["visual_prediction"]["compute_visual_prediction"]
        self.lips_aperture_feature_level = preprocess_config["preprocessing"]["lips_aperture"][
            "feature"
        ]
        self.lips_spreading_feature_level = preprocess_config["preprocessing"]["lips_spreading"][
            "feature"
        ]
        self.use_variance_predictor_visual = model_config["use_variance_predictor_visual"]

    def forward(self, inputs, predictions, no_spectro=False):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            phon_align_targets,
            au_targets,
            _,
            _,
            lips_aperture_targets,
            lips_spreading_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            phon_align_predictions,
            au_predictions,
            postnet_au_predictions,
            lips_aperture_predictions,
            lips_spreading_predictions,
            au_masks,
            _,
            src_masks_noSpectro,
        ) = predictions
    
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        src_masks_noSpectro = ~src_masks_noSpectro

        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        phon_align_targets.requires_grad = False

        if no_spectro or (not torch.any(src_masks_noSpectro)):
            mel_loss = torch.Tensor([0]).long().to(device)
            postnet_mel_loss = torch.Tensor([0]).long().to(device)
            pitch_loss = torch.Tensor([0]).long().to(device)
            energy_loss = torch.Tensor([0]).long().to(device)
            duration_loss = torch.Tensor([0]).long().to(device)
            au_loss = torch.Tensor([0]).long().to(device)
            postnet_au_loss = torch.Tensor([0]).long().to(device)
            lips_aperture_loss = torch.Tensor([0]).long().to(device)
            lips_spreading_loss = torch.Tensor([0]).long().to(device)
        else:
            # Variance Adaptor: AUDIO
            if self.use_variance_predictor["pitch"]:
                if self.pitch_feature_level == "phoneme_level":
                    pitch_predictions = pitch_predictions.masked_select(src_masks_noSpectro)
                    pitch_targets = pitch_targets.masked_select(src_masks_noSpectro)
                elif self.pitch_feature_level == "frame_level":
                    pitch_predictions = pitch_predictions.masked_select(mel_masks)
                    pitch_targets = pitch_targets.masked_select(mel_masks)
                
                pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            else:
                pitch_loss = torch.Tensor([0]).long().to(device)

            if self.use_variance_predictor["energy"]:
                if self.energy_feature_level == "phoneme_level":
                    energy_predictions = energy_predictions.masked_select(src_masks_noSpectro)
                    energy_targets = energy_targets.masked_select(src_masks_noSpectro)
                if self.energy_feature_level == "frame_level":
                    energy_predictions = energy_predictions.masked_select(mel_masks)
                    energy_targets = energy_targets.masked_select(mel_masks)

                energy_loss = self.mse_loss(energy_predictions, energy_targets)
            else:
                energy_loss = torch.Tensor([0]).long().to(device)

            log_duration_predictions = log_duration_predictions.masked_select(src_masks_noSpectro)
            log_duration_targets = log_duration_targets.masked_select(src_masks_noSpectro)

            duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

            if self.compute_visual_prediction and torch.any(au_masks):
                au_masks = ~au_masks
                au_targets = au_targets[:, : au_masks.shape[1], :]
                au_masks = au_masks[:, :au_masks.shape[1]]
                au_targets.requires_grad = False

                lips_aperture_targets.requires_grad = False
                lips_spreading_targets.requires_grad = False

                # Variance Adaptor: VISUAL
                if self.use_variance_predictor_visual["lips_aperture"]:
                    if self.lips_aperture_feature_level == "phoneme_level":
                        lips_aperture_predictions = lips_aperture_predictions.masked_select(src_masks_noSpectro)
                        lips_aperture_targets = lips_aperture_targets.masked_select(src_masks_noSpectro)
                    elif self.lips_aperture_feature_level == "frame_level":
                        lips_aperture_predictions = lips_aperture_predictions.masked_select(au_masks)
                        lips_aperture_targets = lips_aperture_targets.masked_select(au_masks)

                    lips_aperture_loss = self.mse_loss(lips_aperture_predictions, lips_aperture_targets)
                else:
                    lips_aperture_loss = torch.Tensor([0]).long().to(device)

                if self.use_variance_predictor_visual["lips_spreading"]:
                    if self.lips_spreading_feature_level == "phoneme_level":
                        lips_spreading_predictions = lips_spreading_predictions.masked_select(src_masks_noSpectro)
                        lips_spreading_targets = lips_spreading_targets.masked_select(src_masks_noSpectro)
                    if self.lips_spreading_feature_level == "frame_level":
                        lips_spreading_predictions = lips_spreading_predictions.masked_select(au_masks)
                        lips_spreading_targets = lips_spreading_targets.masked_select(au_masks)

                    lips_spreading_loss = self.mse_loss(lips_spreading_predictions, lips_spreading_targets)
                else:
                    lips_spreading_loss = torch.Tensor([0]).long().to(device)

                au_targets = au_targets.masked_select(au_masks.unsqueeze(-1))
                au_predictions = au_predictions.masked_select(au_masks.unsqueeze(-1))

                au_loss = self.mae_loss(au_predictions, au_targets)
                if (postnet_au_predictions is not None):
                    postnet_au_predictions = postnet_au_predictions.masked_select(
                        au_masks.unsqueeze(-1)
                    )

                    postnet_au_loss = self.mae_loss(postnet_au_predictions, au_targets)
                else: 
                    postnet_au_loss = torch.Tensor([0]).long().to(device)
            else:
                au_loss = torch.Tensor([0]).long().to(device)
                postnet_au_loss = torch.Tensor([0]).long().to(device)
                lips_aperture_loss = torch.Tensor([0]).long().to(device)
                lips_spreading_loss = torch.Tensor([0]).long().to(device)

        if self.compute_phon_prediction:
            phon_align_loss = self.cross_entropy_loss(phon_align_predictions, phon_align_targets)
        else:
            phon_align_loss = torch.Tensor([0]).long().to(device)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + phon_align_loss + au_loss + postnet_au_loss + lips_aperture_loss + lips_spreading_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            phon_align_loss,
            au_loss,
            postnet_au_loss,
            lips_aperture_loss,
            lips_spreading_loss,
        )
