# app.py
import gradio as gr
import io
import os
from PIL import Image
import contextlib
import numpy as np
import uuid
import cv2

from inference_painting import do_inf_inpaint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "../autodl-tmp/qwen3_4B"


def save_temp_image(image_array, prefix="img", ext="png"):
    os.makedirs("tmp", exist_ok=True)
    path = f"tmp/{prefix}_{uuid.uuid4().hex}.{ext}"
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    Image.fromarray(image_array).save(path)
    return path

def extract_keywords_from_sentence_llm(sentence):
    """
    Extract two keywords from a sentence using a small LLM.
    """
    prompt = (
        f"You are a photo inpainting assistant. Extract two keywords from the following prompt. "
        f"The first is the object to remove or replace, the second is the new object. "
        f"Only return the two keywords separated by a comma:\n\n{sentence}"
    )

    llm_extract = pipeline(
        "text-generation",
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [{"role": "user", "content": prompt}]
    output = llm_extract(messages, max_new_tokens=1000)

    # 提取 assistant 回复内容
    turns = output[0]["generated_text"]
    assistant_reply = ""
    for m in turns:
        if m["role"] == "assistant":
            assistant_reply = m["content"]
            break

    if not assistant_reply:
        print("❌ No assistant reply found.")
        return sentence, ""

    # 提取 </think> 后内容
    if "</think>" in assistant_reply:
        final_output = assistant_reply.split("</think>")[-1].strip()
    else:
        final_output = assistant_reply.strip()

    print("🧠 Final extracted:", final_output)

    # 提取关键词
    if "," in final_output:
        parts = final_output.split(",", 1)
        return parts[0].strip(), parts[1].strip()
    else:
        print("⚠️ Unexpected format:", final_output)
        return sentence, ""


def create_img(params, mask):
    print("Step:", params["step"])
    print("Guidance Scale:", params["guidance_scale"])
    if mask is not None:
        print("Mask shape:", mask.shape)

    # Extract keywords from sentence
    if "prompt" in params:
        sentence = params["prompt"]
        kw1, kw2 = extract_keywords_from_sentence_llm(sentence)
    else:
        kw1, kw2 = params["kw1"], params["kw2"]
        
    output_image = do_inf_inpaint(
        params["input_path"],
        kw1,
        kw2,
        params["guidance_scale"],
        mask=mask
    )

    return output_image

# Toggles between showing and hiding the sketchpad
def toggle_mask_drawer(show, image):
    if image is None:
        gr.Warning("Please upload an input image before using the brush tool.")
        return (
            gr.update(visible=False),
            gr.update(value="Open brush tool"),
            False
        )

    # Resize mask_canvas to match input image size
    height, width = image.shape[:2]
    return (
        gr.update(visible=not show, value=image, height=height, width=width),
        gr.update(value="Close brush tool" if not show else "Open brush tool"),
        not show
    )

# Prompt input mode change (sentence vs keywords)
def toggle_prompt_components(mode):
    if mode == "Whole sentence":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

# Run generation
def run_pipeline(input_img, mode, sentence, kw1, kw2, steps, guidance, raw_mask):
    if input_img is None:
        gr.Warning("Please upload an input image.")
        return None, "❌ No input image provided."
    
    input_path = save_temp_image(input_img, prefix="input")

    if mode == "Whole sentence":
        params = {"input_path": input_path, "prompt": sentence, "step": steps, "guidance_scale": guidance}
    else:
        params = {"input_path": input_path, "kw1": kw1, "kw2": kw2, "step": steps, "guidance_scale": guidance}

    # Convert RGBA canvas to 0/1 binary mask
    raw_mask = raw_mask['layers']
    mask = None
    if raw_mask is not None and len(raw_mask) > 0: 
        # Convert to grayscale then threshold
        # step 0: Convert to binary mask
        gray = (raw_mask[0].mean(axis=2) > 0).astype(np.uint8)

        # Step 1: Fill small holes — morphological closing (dilation → erosion)
        kernel = np.ones((10, 10), np.uint8)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Step 2: Enlarge the mask a bit — pure dilation
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # Optional: remove tiny noise (small connected components)
        # contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask_clean = np.zeros_like(dilated)
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 100:  # filter small areas
        #         cv2.drawContours(mask_clean, [cnt], -1, color=1, thickness=-1)
        # mask = mask_clean

        # For now, skip noise filtering
        mask = dilated

        num_nonzero = np.count_nonzero(gray)
        if num_nonzero > 0:
            mask = gray
            print("Mask shape:", mask.shape, "Non-zero pixels:", num_nonzero)

    # Capture logs
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("▶️ Starting generation…")
        try:
            out = create_img(params, mask).resize(input_img.shape[:2][::-1])
            print("✅ Generation done.")
        except Exception as e:
            print("❌ Error during generation:", str(e))
            out = None
    logs = buf.getvalue()

    os.remove(input_path)  # Clean up temp input image

    return out, logs

with gr.Blocks() as demo:
    # Hidden variable to track toggle state
    mask_visible = gr.State(False)

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("""
            <div style="position: relative; display: inline-block;" id="image-stack">
            """)
            input_image = gr.Image(
                label="Input image",
                type="numpy",
                show_label=False,
                elem_id="input-img",
                container=False,
            )
            gr.HTML("</div>")
            
            output_image = gr.Image(label="Output image", type="pil")

        with gr.Column(scale=2):
            prompt_mode = gr.Radio(["Whole sentence", "Two keywords"], label="Prompt input mode", value="Whole sentence")
            sentence_input = gr.Textbox(label="Prompt", visible=True)
            keyword1 = gr.Textbox(label="Keyword 1", visible=False)
            keyword2 = gr.Textbox(label="Keyword 2", visible=False)

            step = gr.Slider(1, 200, value=50, step=1, label="Step")
            guidance = gr.Slider(0, 15, value=7, step=1, label="Guidance scale")

            logs = gr.Textbox(label="Logs", interactive=False, lines=8)

            brush_button = gr.Button("Open brush tool")
            mask_canvas = gr.Sketchpad(
                label="Draw mask",
                brush=60,
                type="numpy",
                visible=False
            )

            start_button = gr.Button("Start")

    # Events
    prompt_mode.change(
        fn=toggle_prompt_components,
        inputs=prompt_mode,
        outputs=[sentence_input, keyword1, keyword2]
    )

    brush_button.click(
        fn=toggle_mask_drawer,
        inputs=[mask_visible, input_image],
        outputs=[mask_canvas, brush_button, mask_visible]
    )

    start_button.click(
        fn=run_pipeline,
        inputs=[input_image, prompt_mode, sentence_input, keyword1, keyword2, step, guidance, mask_canvas],
        outputs=[output_image, logs]
    )

demo.launch(share=True)

