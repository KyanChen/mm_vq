import einops
import numpy as np
import torch
from diffusers.models.vae import Encoder, Decoder
from mmengine.model import BaseModule
from typing import Tuple, Optional, List
import torch.nn.functional as F
from mmengine.structures import PixelData
from torch import nn, Tensor

from mmseg.models import EncoderDecoder, accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import ConfigType, SampleList, add_prefix, OptSampleList


@MODELS.register_module()
class VectorQuantizer(BaseModule):
    def __init__(
            self,
            n_e: int,
            vq_embed_dim: int,
            legacy: bool = False,
            beta: float = 0.25,
            with_codebook_reset: bool = False,
            mu=0.99,
            init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.mu = mu

        self.with_codebook_reset = with_codebook_reset
        if with_codebook_reset:
            self.init = False
            self.code_sum = None
            self.code_count = None

    def _tile(self, z_flattened):
        nb_code_x, code_dim = z_flattened.shape
        if nb_code_x < self.n_e:
            n_repeats = (self.n_e + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = z_flattened.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = z_flattened
        return out

    def init_codebook(self, z_flattened):
        out = self._tile(z_flattened)
        self.embedding.weight.data = out[:self.n_e]
        self.code_sum = self.embedding.clone()
        self.code_count = torch.ones(self.n_e, device=self.embedding.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.n_e, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.n_e, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.n_e]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.n_e, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.n_e, self.vq_embed_dim) / self.code_count.view(self.n_e, 1)

        self.embedding.weight.data = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = einops.rearrange(z, 'b h w c -> (b h w) c')
        assert z_flattened.shape[-1] == self.vq_embed_dim

        if self.training and self.with_codebook_reset and not self.init:
            self.init_codebook(z_flattened)

        min_encoding_indices, z_q = self.quantize(z_flattened)
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.training and self.with_codebook_reset:
            perplexity = self.update_codebook(z_flattened, min_encoding_indices)
        else:
            perplexity = self.compute_perplexity(min_encoding_indices)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        return z_q, loss, (perplexity, min_encoding_indices)

    def quantize(self, z_flattened):

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)
        z_q = self.embedding(min_encoding_indices)

        # preserve gradients
        z_q = z_flattened + (z_q - z_flattened).detach()
        return min_encoding_indices, z_q

    def dequantize(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.vq_embed_dim,)).contiguous()
        return z_q



@MODELS.register_module()
class VQVAEModel(BaseModule):
    def __init__(
        self,
        quantizer,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        norm_num_groups: int = 32,
        vq_embed_dim: Optional[int] = None,
        norm_type: str = "group",  # group, spatial
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)

        self.quantizer = MODELS.build(quantizer)

        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize: bool = False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, loss, (perplexity, min_encoding_indices) = self.quantizer(h)
        else:
            quant, loss, (perplexity, min_encoding_indices) = h, 0, (0, 0)
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2)
        return dict(sample=dec, commit_loss=loss, perplexity=perplexity, min_encoding_indices=min_encoding_indices)

    def forward(self, x):
        h = self.encode(x)
        dec = self.decode(h)
        return dec


@MODELS.register_module()
class VQVAEEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 **kwargs
                 ):
        super().__init__(backbone=backbone, decode_head=decode_head, **kwargs)

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # dict(sample=dec, commit_loss=loss, perplexity=perplexity, min_encoding_indices=min_encoding_indices)
        return_dict = self.backbone(inputs)
        losses = dict(commit_loss=return_dict['commit_loss'], perplexity=return_dict['perplexity'])
        inputs = return_dict['sample']
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        return_dict = self.backbone(inputs)
        inputs = return_dict['sample']
        seg_logits = self.decode_head.predict(inputs, batch_img_metas, self.test_cfg)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples


class PseudoDecodeHead(BaseModule):
    def __init__(self,
                 threshold=0.5,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.threshold = threshold
        self.loss_decode = MODELS.build(loss_decode)

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        return inputs

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(inputs)

        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        seg_label = torch.stack(gt_semantic_segs, dim=0)

        losses = dict()

        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)

        losses['ce_loss'] = self.loss_decode(seg_logits, seg_label, ignore_index=self.ignore_index)
        losses['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict], test_cfg: ConfigType) -> Tensor:
        seg_logits = self.forward(inputs)
        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits