"""
One-tower backbone.

Phase 1 wraps the existing `renaissance.modules.one_tower_encoder.
OneTowerEncoder` behind the `Backbone` protocol — the encoder internals
are untouched so the Phase 0 contract tests stay valid. Later phases may
refactor the wrapped internals; callers only ever see `Backbone`.
"""

from renaissance.modules.one_tower_encoder import OneTowerEncoder

from .base import Backbone, EncoderOutput


class OneTowerBackbone(Backbone):
    """Shared HF backbone for both modalities, with `single` or `double`
    CLS pooling."""

    def __init__(self, config):
        super().__init__()
        self._pooler_type = config["pooler_type"]
        self.encoder = OneTowerEncoder(
            config,
            config["image_size"],
            config["max_text_len"],
            fine_tune=False,
            test_only=False,
        )
        self._hidden_size = self.encoder.get_hidden_size()

    @property
    def pooled_dim(self) -> int:
        if self._pooler_type == "single":
            return self._hidden_size
        return 2 * self._hidden_size

    @property
    def token_dim(self) -> int:
        return self._hidden_size

    @property
    def text_hidden_size(self) -> int:
        return self._hidden_size

    @property
    def image_hidden_size(self) -> int:
        return self._hidden_size

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
        return self.encoder.forward_text(batch)

    def adjust_type_embeds_for_nlvr2(self):
        """Passthrough until Phase 3 moves the dual-image trick into the
        NLVR2 task."""
        self.encoder.adjust_type_embeds_for_nlvr2()
