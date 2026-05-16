"""
Two-tower backbone.

Phase 1 wraps the existing `renaissance.modules.two_tower_encoder.
TwoTowerEncoder` behind the `Backbone` protocol — separate HF text and
vision encoders fused by the LXMERT cross-modal module. Encoder internals
are untouched (Phase 0 contract tests stay valid); callers only see
`Backbone`.
"""

from renaissance.modules.two_tower_encoder import TwoTowerEncoder

from .base import Backbone, EncoderOutput


class TwoTowerBackbone(Backbone):
    """Separate text + vision HF encoders, projected to a shared width and
    fused by parallel cross-attention. The pooled feature is the
    concatenation of the fused text and image CLS poolers."""

    def __init__(self, config):
        super().__init__()
        self._cross_hidden = config["cross_layer_hidden_size"]
        self.encoder = TwoTowerEncoder(config, fine_tune=False, test_only=False)

    @property
    def pooled_dim(self) -> int:
        # cls_feats = concat(text_pooled, image_pooled), each cross_hidden.
        return 2 * self._cross_hidden

    @property
    def token_dim(self) -> int:
        # Fusion projects both streams to cross_layer_hidden_size.
        return self._cross_hidden

    @property
    def text_hidden_size(self) -> int:
        return self.encoder.text_transformer_hidden_size

    @property
    def image_hidden_size(self) -> int:
        return self.encoder.image_encoder_hidden_size

    def forward(
        self,
        batch,
        *,
        mask_text: bool = False,
        mask_image: bool = False,
        image_token_type_idx: int = 1,
    ) -> EncoderOutput:
        ret = self.encoder(
            batch,
            mask_text=mask_text,
            mask_image=mask_image,
            image_token_type_idx=image_token_type_idx,
        )
        return EncoderOutput(
            pooled=ret["cls_feats"],
            text_tokens=ret["text_feats"],
            image_tokens=ret["image_feats"],
            text_ids=ret.get("text_ids"),
            text_labels=ret.get("text_labels"),
            text_masks=ret.get("text_masks"),
        )

    def encode_text_only(self, batch):
        return self.encoder.text_transformer(**batch).last_hidden_state

    def adjust_type_embeds_for_nlvr2(self):
        """Passthrough until Phase 3 moves the dual-image trick into the
        NLVR2 task."""
        self.encoder.adjust_type_embeds_for_nlvr2()
