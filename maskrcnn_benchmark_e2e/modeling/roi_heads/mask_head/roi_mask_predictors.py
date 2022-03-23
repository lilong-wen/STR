# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark_e2e.layers import Conv2d, ConvTranspose2d
import torch
from torch import nn
from torch.nn import functional as F

# from .roi_seq_predictors import make_roi_seq_predictor
from .feature_align import feature_align_for_x_and_target
from .target_generator import WordImageGenerator
from .matcher import MatchERT

class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, global_features, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')



class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            if cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                num_inputs = dim_reduced + 1
            elif cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
                num_inputs = dim_reduced * 2
            else:
                num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class CharMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(CharMaskRCNNC4Predictor, self).__init__()
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes = 1
        char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            if cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                num_inputs = dim_reduced + 1
            elif cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
                num_inputs = dim_reduced * 2
            else:
                num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.MODEL.CHAR_MASK_ON:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.char_mask_fcn_logits = Conv2d(dim_reduced, char_num_classes, 1, 1, 0)
        else:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x), self.char_mask_fcn_logits(x)


class SeqCharMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(SeqCharMaskRCNNC4Predictor, self).__init__()
        # num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes = 1
        char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        self.target_generator = WordImageGenerator(cfg.CHAR_DATA.PATH, 720, 32, 32)
        self.feature_align_for_x_and_target = feature_align_for_x_and_target()
        self.matcherERT = MatchERT()
        self.class_loss = BinaryCrossEntropyWithLogits()        

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            if cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                num_inputs = dim_reduced + 1
            elif cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or 'ATTENTION_CHANNEL' in cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION:
                num_inputs = dim_reduced * 2
            else:
                num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.MODEL.CHAR_MASK_ON:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.char_mask_fcn_logits = Conv2d(dim_reduced, char_num_classes, 1, 1, 0)
            # self.seq = make_roi_seq_predictor(cfg, dim_reduced)
        else:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.normal(param)

    def forward(self, x, decoder_targets=None, word_targets=None):
        x_origin = torch.clone(x)
        x = F.relu(self.conv5_mask(x))
        
        if self.training:
            word_targets_imgs =  torch.from_numpy(self.target_generator(decoder_targets)).float().to(decoder_targets.device)

            # loss_seq_decoder = self.seq(
            #     x, decoder_targets=decoder_targets, word_targets=word_targets
            # )

            word_targets_align, x_align_anchor = self.feature_align_for_x_and_target(word_targets_imgs, x_origin)
            word_targets_positive = word_targets_align[0::2]
            word_targets_negtive = word_targets_align[1::2]
            p_logits = self.matcherERT(x_align_anchor, word_targets_positive)
            n_logits = self.matcherERT(x_align_anchor, word_targets_negtive)
            logits = torch.cat([p_logits, n_logits], 0)
            bsize = logits.size(0)
            labels = logits.new_ones(logits.size())
            labels[(bsize//2):] = 0
            loss_seq_decoder = self.class_loss(logits, None, labels).mean()

            return (
                self.mask_fcn_logits(x),
                self.char_mask_fcn_logits(x),
                loss_seq_decoder,
            )
        else:
            decoded_chars, decoded_scores, detailed_decoded_scores = self.seq(
                x, use_beam_search=True
            )
            return (
                self.mask_fcn_logits(x),
                self.char_mask_fcn_logits(x),
                decoded_chars,
                decoded_scores,
                detailed_decoded_scores,
            )

class SeqMaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(SeqMaskRCNNC4Predictor, self).__init__()
        num_classes = 1
        # char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            if cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                num_inputs = dim_reduced + 1
            elif cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
                num_inputs = dim_reduced * 2
            else:
                num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.SEQUENCE.SEQ_ON:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.seq = make_roi_seq_predictor(cfg, dim_reduced)
        else:
            self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, decoder_targets=None, word_targets=None):
        x = F.relu(self.conv5_mask(x))
        if self.training:
            loss_seq_decoder = self.seq(
                x, decoder_targets=decoder_targets, word_targets=word_targets
            )
            return (
                self.mask_fcn_logits(x),
                loss_seq_decoder,
            )
        else:
            decoded_chars, decoded_scores, detailed_decoded_scores = self.seq(
                x, use_beam_search=True
            )
            return (
                self.mask_fcn_logits(x),
                decoded_chars,
                decoded_scores,
                detailed_decoded_scores,
            )

class SeqRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(SeqRCNNC4Predictor, self).__init__()
        num_classes = 1
        # char_num_classes = cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            if cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'CAT':
                num_inputs = dim_reduced + 1
            elif cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'MIX' or cfg.MODEL.ROI_MASK_HEAD.MIX_OPTION == 'ATTENTION_CHANNEL':
                num_inputs = dim_reduced * 2
            else:
                num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        if cfg.SEQUENCE.SEQ_ON:
            # self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)
            self.seq = make_roi_seq_predictor(cfg, dim_reduced)
        # else:
        #     self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x, decoder_targets=None, word_targets=None):
        x = F.relu(self.conv5_mask(x))
        if self.training:
            loss_seq_decoder = self.seq(
                x, decoder_targets=decoder_targets, word_targets=word_targets
            )
            return (
                None,
                loss_seq_decoder,
            )
        else:
            decoded_chars, decoded_scores, detailed_decoded_scores = self.seq(
                x, use_beam_search=True
            )
            return (
                None,
                decoded_chars,
                decoded_scores,
                detailed_decoded_scores,
            )

_ROI_MASK_PREDICTOR = {
    "MaskRCNNC4Predictor": MaskRCNNC4Predictor,
    "CharMaskRCNNC4Predictor": CharMaskRCNNC4Predictor,
    "SeqCharMaskRCNNC4Predictor": SeqCharMaskRCNNC4Predictor,
    "SeqMaskRCNNC4Predictor": SeqMaskRCNNC4Predictor,
    "SeqRCNNC4Predictor": SeqRCNNC4Predictor,
}


def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
