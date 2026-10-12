# BLIP-2 with LAVIS

The BLIP-2 matching head and the BLIP-2 feature extractor are run with the LAVIS implementation used in the
original evaluation (`blip2_image_text_matching` / `pretrain` and `blip2_feature_extractor` / `pretrain`), not
with the Hugging Face port. LAVIS pins its own dependencies, so it lives in a separate environment:

```
uv venv lavis-env --python 3.10
uv pip install --python lavis-env/Scripts/python.exe torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python lavis-env/Scripts/python.exe salesforce-lavis pandas scipy tqdm pytest
```

Run the BLIP-2 parts of the pipeline with that interpreter and the package on the path:

```
set PYTHONPATH=<path to pipeline>
lavis-env\Scripts\python.exe scripts\03b_itm_benchmark.py BLIP2
lavis-env\Scripts\python.exe scripts\07_itm_pools.py BLIP2
lavis-env\Scripts\python.exe scripts\20_coco_embed.py BLIP2
```

Everything else runs in `.venv`.
