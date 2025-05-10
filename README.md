# Text-Guided Image Inpainting

This repo allows you to perform **image editing via text prompts**, or support user-defined masking area and generation prompts.

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
```

To download the weights for SlipSeg, you can use the following command:

```bash
cd clipseg
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
cd ..
```

Then in `masking_inpainting.py`, you need to set the diffusion model load path on line **141**, where I defined:
```python
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    '/nvme0n1/xmy/stable-diffusion-2-inpainting', # Change this to your own local dir
    revision='fp16', 
    torch_dtype=torch.float16,
    use_safetensors=True
).to(device)
```

The small `ViT-B 16` model will be automatically downloaded when you run the code from HuggingFace. If you meet the error of unable to connect to huggingface, try this:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python masking_inpainting.py
```

## Running
You can modify the `masking_inpainting.py` file to set your own image path, mask path, and text prompt. See `main` function for details. Happy trying!
