# Text-Guided Image Inpainting

This repo allows you to perform **image editing via text prompts**, or support user-defined masking area and generation prompts.

## TaskList

- [x] Text-guided image inpainting
- [x] User-defined masking area
- [ ] Complete WebUI support (Check TODOs in code)
- [ ] Train the clip model for mask prediction
- [ ] Design a metric for mask prediction quality (I suggest miou for quality, k-means for clustering)
- [ ] Do ablation study
- [ ] Finish the report

You may download the **Magic Brush** dataset using the the codes in `dataset/download.py`, and extract the dataset using `dataset/extract.py`. The dataset will be formatted into:


```
MagicBrush/
├── train
│   ├── 00000
│   │   ├── instructions.txt (with inpaint instructions)
│   │   ├── mask_img.png (img with masked out areas)
│   │   ├── source_img.png (original image)
│   │   └── target_img.png (image with masked areas filled)
│   └── ...
├── test
│   └── ...
```


## Environment Setup
First of all, set up a new conda environment:
```bash
conda create -n paint python=3.10 -y
conda activate paint
```

Then install the required packages:
```bash
pip install torch==2.6.0
pip install torchvision==0.21.0
pip install -r requirements.txt
```

I advise that you also install flash-attn to speed up inference. If you installed all prior dependencies following this guide, then you can choose to install via this method:
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.1.post4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
otherwise, you can install via `pip`, which might be slow and cause trouble:
```bash
pip install flash-attn
```
if you really have trouble installing flash-attn, simply disable it in `VLinference.py`.


Then we install CLIP module:
```bash
cd CLIP
pip install -e .
cd ..
```

## Model and Weights
In the current impl, I used the `stable-diffusion-2-inpainting` model, also with `ViT-B 16` and some weights for SlipSeg. For the large model, I recommend using `modelscope` for downloading:

```bash
pip install modelscope # If you do not have modelscpope installed
modelscope download --model stabilityai/stable-diffusion-2-inpainting --local_dir <your local dir>
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir <your local dir>
```

To download the weights for SlipSeg, you can use the following command:

```bash
cd clipseg
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
cd ..
```

Then in `masking_inpainting.py`, you need to set the diffusion model load path on line **126**, where I defined:
```python
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    '/nvme0n1/xmy/stable-diffusion-2-inpainting', # Change this to your own local dir
    revision='fp16', 
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)
```

Also, in `VLinference.py`, you need to set the model path on line **6**:
```python
model_path="/nvme0n1/xmy/Qwen2.5-VL-7B-Instruct" # Change this to your own local dir
```

The small `ViT-B 16` model will be automatically downloaded when you run the code from HuggingFace. If you meet the error of unable to connect to huggingface, try this:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python masking_inpainting.py
```

## Running
You can modify the `inference_inpainting.py` file to set your own image path, mask path, and text prompt. See `main` function for details. Happy trying!

## More
Checkout the instruct-pix2pix folder for newly trained models for inpainting!
