"""One implementation of the BLIP-2 and FLAVA image-text matching (ITM) heads.

The same code scores the benchmark pairs (scripts/03b_itm_benchmark.py) and the judged pool (scripts/07_itm_pools.py),
so the two numbers of a system are comparable.

BLIP-2: LAVIS `blip2_image_text_matching` / `pretrain` (the implementation of the original evaluation), loaded with
        `load_model_and_preprocess` and its own image and text processors, `match_head="itm"`, softmax over the two
        ITM logits. Requires the `salesforce-lavis` package (separate environment; see scripts/README_lavis.md).
FLAVA:  facebook/flava-full, FlavaForPreTraining ITM head, fp32, softmax over the two ITM logits.
A pair is a match when p(match) > 0.5. Images that cannot be read or are shorter than 32 px on a side get no
decision (None) and are excluded from every accuracy.
"""
from functools import lru_cache
import torch
from PIL import Image
from .paths import IMAGES_DIR

MIN_SIDE = 32
THRESHOLD = 0.5
# BLIP: LAVIS `blip_image_text_matching` / `base` (ViT-B, 384 px, the COCO retrieval checkpoint whose ITM head re-ranks
# in the published BLIP setting), `match_head="itm"`, softmax over the two ITM logits.
HEAD_IDS = {"BLIP": "lavis:blip_image_text_matching/base", "BLIP2": "lavis:blip2_image_text_matching/pretrain",
            "FLAVA": "facebook/flava-full"}


@lru_cache(maxsize=512)
def load_image(image_id):
    try:
        im = Image.open(IMAGES_DIR / f"{int(image_id)}.jpg").convert("RGB")
    except Exception:
        return None
    if min(im.size) < MIN_SIDE:
        return None
    return im


class MatchingHead:
    def __init__(self, model, device=None, batch_size=16):
        if model not in HEAD_IDS:
            raise ValueError(model)
        self.model = model
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model in ("BLIP", "BLIP2"):
            from lavis.models import load_model_and_preprocess
            name, model_type = HEAD_IDS[model].split(":")[1].split("/")
            self.net, self.vis_proc, self.txt_proc = load_model_and_preprocess(
                name=name, model_type=model_type, is_eval=True, device=torch.device(self.device))
            self.net = self.net.eval()
            return
        from transformers import FlavaForPreTraining, FlavaProcessor
        self.proc = FlavaProcessor.from_pretrained(HEAD_IDS[model])
        self.net = FlavaForPreTraining.from_pretrained(HEAD_IDS[model])
        self.net.config.use_mim = False
        self.net = self.net.to(self.device).eval()

    @torch.no_grad()
    def _match_probs(self, images, captions):
        if self.model == "BLIP":
            # LAVIS BlipITM.forward(match_head="itm") pads the texts itself ("longest"), so pairs batch directly.
            img = torch.stack([self.vis_proc["eval"](im) for im in images]).to(self.device)
            txt = [self.txt_proc["eval"](c) for c in captions]
            logits = self.net({"image": img, "text_input": txt}, match_head="itm")
        elif self.model == "BLIP2":
            # LAVIS Blip2ITM.forward(match_head="itm") without padding only accepts equal-length texts; this is the
            # same computation with padded, attention-masked text so that pairs can be batched.
            net = self.net
            img = torch.stack([self.vis_proc["eval"](im) for im in images]).to(self.device)
            txt = [self.txt_proc["eval"](c) for c in captions]
            with net.maybe_autocast():
                image_embeds = net.ln_vision(net.visual_encoder(img))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=img.device)
            text = net.tokenizer(txt, padding=True, truncation=True, max_length=net.max_txt_len, return_tensors="pt").to(img.device)
            query_tokens = net.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=img.device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            out = net.Qformer.bert(text.input_ids, query_embeds=query_tokens, attention_mask=attention_mask,
                                   encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True)
            logits = net.itm_head(out.last_hidden_state[:, : query_tokens.size(1), :]).mean(dim=1)
        else:
            inp = self.proc(text=captions, images=images, return_codebook_pixels=True, padding=True,
                            return_tensors="pt").to(self.device)
            logits = self.net(**inp, return_loss=False).itm_logits
        return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()

    def match_probs(self, pairs, progress=None, image_loader=None):
        """pairs: sequence of (image_id, caption). Returns p(match) per pair, None where the image is unusable.
        image_loader(image_id) -> PIL image or None; default: the SVO-Probes images (load_image)."""
        image_loader = image_loader or load_image
        pairs = list(pairs)
        out = [None] * len(pairs)
        idx, imgs, caps = [], [], []

        def flush():
            if idx:
                for i, p in zip(idx, self._match_probs(imgs, caps)):
                    out[i] = float(p)
                idx.clear()
                imgs.clear()
                caps.clear()

        it = enumerate(pairs)
        if progress is not None:
            it = progress(it, total=len(pairs))
        for i, (img, cap) in it:
            im = image_loader(img)
            if im is None:
                continue
            idx.append(i)
            imgs.append(im)
            caps.append(str(cap))
            if len(idx) >= self.batch_size:
                flush()
        flush()
        return out

    def decisions(self, pairs, progress=None):
        return [None if p is None else bool(p > THRESHOLD) for p in self.match_probs(pairs, progress)]
