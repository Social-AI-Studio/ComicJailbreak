from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = torch.bfloat16
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28


@dataclass(frozen=True)
class LocalModelBackend:
    family: str
    model: object
    processor: object
    process_input: Callable


def _move_to_device(inputs):
    return inputs.to(DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)


def _load_model(model_cls, model_id: str, *, use_flash_attention: bool = False):
    kwargs = {
        "device_map": "auto",
        "dtype": DEFAULT_DTYPE,
    }
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    return model_cls.from_pretrained(model_id, **kwargs).eval()


def _load_processor(model_id: str, **kwargs):
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def _process_chat_template(processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return _move_to_device(inputs)


def _extract_first_image(messages) -> Image.Image | None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                return Image.open(item["image"])
    return None


def detect_model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen2.5" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    if "llama" in lowered:
        return "llama"
    if "qwen3.5" in lowered:
        return "qwen3.5"
    if "qwen3" in lowered:
        return "qwen3"
    raise NotImplementedError(f"Unsupported local model: {model_id}")


def get_local_backend(model_id: str) -> LocalModelBackend:
    family = detect_model_family(model_id)

    if family == "qwen":
        return LocalModelBackend(
            family=family,
            model=qwen_model(model_id),
            processor=qwen_processor(model_id),
            process_input=qwen_process_input,
        )
    if family == "gemma":
        return LocalModelBackend(
            family=family,
            model=gemma_model(model_id),
            processor=gemma_processor(model_id),
            process_input=gemma_process_input,
        )
    if family == "llama":
        return LocalModelBackend(
            family=family,
            model=llama_model(model_id),
            processor=llama_processor(model_id),
            process_input=llama_process_input,
        )
    if family == "qwen3.5":
        return LocalModelBackend(
            family=family,
            model=qwen3_5_model(model_id),
            processor=qwen3_5_processor(model_id),
            process_input=qwen3_5_process_input,
        )
    if family == "qwen3":
        return LocalModelBackend(
            family=family,
            model=qwen3_model(model_id),
            processor=qwen3_processor(model_id),
            process_input=qwen3_process_input,
        )
    raise NotImplementedError(f"Unsupported local model: {model_id}")


def qwen_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    return _load_model(
        Qwen2_5_VLForConditionalGeneration,
        model_id
    )


def qwen_processor(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    return _load_processor(
        model_id,
        min_pixels=QWEN_MIN_PIXELS,
        max_pixels=QWEN_MAX_PIXELS,
    )


def qwen_helper(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    return qwen_model(model_id), qwen_processor(model_id)


def qwen_process_input(processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return _move_to_device(inputs)


def gemma_model(model_id="google/gemma-3-12b-it"):
    return _load_model(Gemma3ForConditionalGeneration, model_id)


def gemma_processor(model_id="google/gemma-3-12b-it"):
    return _load_processor(model_id)


def gemma_helper(model_id="google/gemma-3-12b-it"):
    return gemma_model(model_id), gemma_processor(model_id)


def gemma_process_input(processor, messages):
    return _process_chat_template(processor, messages)


def llama_model(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    return _load_model(MllamaForConditionalGeneration, model_id)


def llama_processor(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    return _load_processor(model_id)


def llama_helper(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    return llama_model(model_id), llama_processor(model_id)


def llama_process_input(processor, messages):
    image = _extract_first_image(messages)
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return _move_to_device(inputs)


def qwen3_model(model_id="Qwen/Qwen3-VL-8B-Instruct"):
    return _load_model(Qwen3VLForConditionalGeneration, model_id)


def qwen3_processor(model_id="Qwen/Qwen3-VL-8B-Instruct"):
    return _load_processor(model_id)


def qwen3_helper(model_id="Qwen/Qwen3-VL-8B-Instruct"):
    return qwen3_model(model_id), qwen3_processor(model_id)


def qwen3_process_input(processor, messages):
    return _process_chat_template(processor, messages)


def qwen3_5_model(model_id="Qwen/Qwen3.5-9B"):
    return _load_model(Qwen3_5ForConditionalGeneration, model_id)


def qwen3_5_processor(model_id="Qwen/Qwen3.5-9B"):
    return _load_processor(model_id)


def qwen3_5_helper(model_id="Qwen/Qwen3.5-9B"):
    return qwen3_5_model(model_id), qwen3_5_processor(model_id)


def qwen3_5_process_input(processor, messages):
    return _process_chat_template(processor, messages)


def image_add_AsD(img_path, out_path="./defense/AsD/temp.png", return_img=False):
    asd_image = Image.open("./defense/AsD/prompt.png").convert("RGB")
    base_image = Image.open(img_path).convert("RGB")

    image_width, image_height = base_image.size
    image_area = image_width * image_height
    target_area = image_area * 0.25

    aspect_ratio = asd_image.width / asd_image.height
    target_width = int((target_area * aspect_ratio) ** 0.5)
    target_height = int(target_width / aspect_ratio)

    overlay = asd_image.resize((target_width, target_height)).convert("RGB")
    overlay_alpha = overlay.copy()
    overlay_alpha.putalpha(int(0.7 * 255))

    max_x = image_width - target_width
    max_y = image_height - target_height
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)
    base_image.paste(overlay_alpha, (random_x, random_y), mask=overlay_alpha)

    if return_img:
        return base_image

    base_image.save(out_path)


__all__ = [
    "LocalModelBackend",
    "detect_model_family",
    "get_local_backend",
    "gemma_helper",
    "gemma_model",
    "gemma_process_input",
    "gemma_processor",
    "image_add_AsD",
    "llama_helper",
    "llama_model",
    "llama_process_input",
    "llama_processor",
    "qwen3_5_helper",
    "qwen3_5_model",
    "qwen3_5_process_input",
    "qwen3_5_processor",
    "qwen3_helper",
    "qwen3_model",
    "qwen3_process_input",
    "qwen3_processor",
    "qwen_helper",
    "qwen_model",
    "qwen_process_input",
    "qwen_processor",
]
