python edit_cli.py --input [YOUR_IMG_PATH] --output imgs/output.jpg --edit "EDIT INSTRUCTION" --ckpt checkpoints/MagicBrush-epoch-52-step-4999.ckpt
 
# Our model is fine-tuned for fine-grained real image editing. Therefore, it may not perform well on style transfer or other editings on large region.
# e.g. python edit_cli.py --input imgs/example.jpg --output imgs/output.jpg --edit "let him wear a pair of sun glasses" --ckpt checkpoints/MagicBrush-epoch-52-step-4999.ckpt
