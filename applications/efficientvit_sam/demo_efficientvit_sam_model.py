import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image
import base64
import google.generativeai as genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)


def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image


def show_anns(anns) -> None:
    """
    Visualize annotations as colored masks over the current matplotlib figure.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    """
    Overlay a binary mask on top of the original image.
    """
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * 0.5
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def draw_bbox(
    image: np.ndarray,
    bbox: list[list[int]],
    color: str | list[str] = "g",
    linewidth=1,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def draw_scatter(
    image: np.ndarray,
    points: list[list[int]],
    color: str | list[str] = "g",
    marker="*",
    s=10,
    ew=0.25,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def get_mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Given a binary mask, return the bounding box (left, top, right, bottom).
    """
    ys, xs = np.where(mask)
    top, left = ys.min(), xs.min()
    bottom, right = ys.max(), xs.max()
    return left, top, right, bottom


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop the image by the given bounding box: (left, top, right, bottom).
    """
    (left, top, right, bottom) = bbox
    return image[top : bottom + 1, left : right + 1]


def draw_text(
    image: np.ndarray, 
    text: str, 
    org=(10, 30), 
    color=(255, 0, 0), 
    font_scale=1.0, 
    thickness=2
) -> np.ndarray:
    """
    Draws 'text' on the image at position 'org'.
    """
    image_copy = image.copy()
    cv2.putText(
        image_copy, 
        text, 
        org, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        color, 
        thickness, 
        lineType=cv2.LINE_AA
    )
    return image_copy


def gemini_annotate_image(image: np.ndarray) -> dict:
    """
    Sends a sub-image (np.ndarray) to Gemini for annotation.
    Returns a dict with the annotation result.
    """

    # 1. Configure your Google GenAI
    genai.configure(api_key="AIzaSyCSZz1Ig-fLZXPkDh-dNwBCSi7sdc3AaZk")

    # 2. Encode the np.ndarray as JPEG in memory
    success, encoded_img = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image to .jpg format.")

    encoded_bytes = encoded_img.tobytes()
    encoded_base64_str = base64.b64encode(encoded_bytes).decode('utf-8')

    # 3. Create your model reference and prompt
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = "Extract label text and predict the image data exactly in two to three words. No extra content please."

    # 4. Send the prompt along with the Base64 image
    response = model.generate_content(
        [
            {
                'mime_type': 'image/jpeg',
                'data': encoded_base64_str
            },
            prompt
        ]
    )

    # 5. Return Gemini's annotation response
    return {
        "status": "ok",
        "annotation": response.text
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--image_path", type=str, default="assets/fig/1.jpg")
    parser.add_argument("--output_path", type=str, default="assets/1.jpg")

    parser.add_argument("--mode", type=str, default="all", choices=["point", "box", "all"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--box", type=str, default=None)

    # EfficientViTSamAutomaticMaskGenerator args
    parser.add_argument("--pred_iou_thresh", type=float, default=0.85)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--min_mask_region_area", type=float, default=600)

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # build model
    efficientvit_sam = create_efficientvit_sam_model(args.model, True, args.weight_url).cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
    )

    # load image
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    annotated_image = raw_image.copy()
    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    tmp_file = f".tmp_{time.time()}.png"

    if args.mode == "all":
        # 1. Generate the masks automatically
        masks = efficientvit_mask_generator.generate(raw_image)

        # 2. Visualize them on the main image (as before)
        plt.figure(figsize=(20, 20))
        plt.imshow(raw_image)
        show_anns(masks)
        plt.axis("off")
        plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close()

        # 3. Additionally, crop out each mask and send to Gemini
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        # Prepare a directory for the labeled subimages
        crop_dir = os.path.join(os.path.dirname(args.output_path), "mask_crops")
        os.makedirs(crop_dir, exist_ok=True)

        for i, ann in enumerate(sorted_masks):
            if i >= 30:
              break
              
            segmentation = ann["segmentation"].astype(np.uint8)  # shape: [H, W]
            # Get bounding box for this particular mask
            left, top, right, bottom = get_mask_bounding_box(segmentation)

            # Crop the original image by that box
            sub_image = crop_image(raw_image, (left, top, right, bottom))

            # 4. Send each sub-image to Gemini for annotation
            gemini_result = gemini_annotate_image(sub_image)
            annotation = gemini_result["annotation"]

            # 5. (Optional) Draw that annotation text on the sub-image
            sub_image_labeled = draw_text(
                sub_image, 
                text=annotation, 
                org=(10, 30), 
                color=(0, 255, 0),  # Green text
                font_scale=1.0, 
                thickness=2
            )

            # 6. Save or log the Gemini result
            print(f"[Gemini] Mask {i} bounding box: {(left, top, right, bottom)}")
            print(f"[Gemini] Annotation result: {annotation}")

            # Save each labeled region
            cv2.rectangle(
                annotated_image,
                (left, top),
                (right, bottom),
                color=(0, 255, 0),  # green box
                thickness=2
            )
            # Draw the label text just above the box
            text_position = (left, max(top - 10, 20))
            annotated_image = draw_text(
                annotated_image,
                text=annotation,
                org=text_position,
                color=(0, 255, 0),  # green text
                font_scale=0.3,
                thickness=4
            )
        
        Image.fromarray(annotated_image).save(args.output_path)
        print(f"Annotated image saved to: {args.output_path}")

    elif args.mode == "point":
        # If no point is specified, default to center of the image
        args.point = yaml.safe_load(f"[[{W // 2},{H // 2},{1}]]" if args.point is None else args.point)
        point_coords = [(x, y) for x, y, _ in args.point]
        point_labels = [l for _, _, l in args.point]

        efficientvit_sam_predictor.set_image(raw_image)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=args.multimask,
        )

        plots = []
        for binary_mask in masks:
            overlay = draw_binary_mask(raw_image, binary_mask, (0, 0, 255))
            scatter_img = draw_scatter(
                overlay,
                point_coords,
                color=["g" if l == 1 else "r" for l in point_labels],
                s=10,
                ew=0.25,
                tmp_name=tmp_file,
            )
            plots.append(scatter_img)

        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)

    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        efficientvit_sam_predictor.set_image(raw_image)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(args.box),
            multimask_output=args.multimask,
        )
        plots = []
        for binary_mask in masks:
            overlay = draw_binary_mask(raw_image, binary_mask, (0, 0, 255))
            bbox_img = draw_bbox(
                overlay,
                [args.box],
                color="g",
                tmp_name=tmp_file,
            )
            plots.append(bbox_img)

        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
