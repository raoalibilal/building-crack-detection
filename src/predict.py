"""Building Crack Inspector: detect and locate cracks in a wall / concrete photo.

Usage:
    python src/predict.py photo.jpg
Saves photo_result.jpg with cracks highlighted in red.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_IMG, SEG_IMG = 224, 256

clf = tf.keras.models.load_model(os.path.join(ROOT, "models", "crack_classifier.keras"), compile=False)
unet = tf.keras.models.load_model(os.path.join(ROOT, "models", "crack_unet.keras"), compile=False)


def segment_full(img):
    """img: HxWx3 uint8. Returns HxW crack probability map (big photos are tiled)."""
    h, w = img.shape[:2]
    if max(h, w) <= 384:
        p = unet.predict(tf.image.resize(img[None].astype("float32"), (SEG_IMG, SEG_IMG)), verbose=0)[0]
        return tf.image.resize(p, (h, w)).numpy()[..., 0]
    T = SEG_IMG
    H, W = int(np.ceil(h / T)) * T, int(np.ceil(w / T)) * T
    pad = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode="reflect").astype("float32")
    tiles = pad.reshape(H // T, T, W // T, T, 3).transpose(0, 2, 1, 3, 4).reshape(-1, T, T, 3)
    p = unet.predict(tiles, batch_size=16, verbose=0)[..., 0]
    return p.reshape(H // T, W // T, T, T).transpose(0, 2, 1, 3).reshape(H, W)[:h, :w]


def inspect(path):
    img = np.array(Image.open(path).convert("RGB")).astype("uint8")
    prob = float(clf.predict(tf.image.resize(img[None].astype("float32"), (CLS_IMG, CLS_IMG)), verbose=0)[0, 0])
    mask = segment_full(img) > 0.5
    area = float(mask.mean() * 100)
    crack = prob > 0.5 or area > 0.5
    severity = "None" if not crack else ("Minor" if area < 1 else "Moderate" if area < 3 else "Severe")
    over = img.copy()
    over[mask] = (0.35 * over[mask] + 0.65 * np.array([255, 0, 0])).astype("uint8")
    return {"crack": crack, "confidence": prob if crack else 1 - prob, "area_pct": area, "severity": severity}, over


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/predict.py photo.jpg")
    r, over = inspect(sys.argv[1])
    out = os.path.splitext(sys.argv[1])[0] + "_result.jpg"
    Image.fromarray(over).save(out)
    print(f"Result     : {'CRACK DETECTED' if r['crack'] else 'NO CRACK'}")
    print(f"Confidence : {r['confidence']:.1%}")
    print(f"Crack area : {r['area_pct']:.2f}%")
    print(f"Severity   : {r['severity']}")
    print(f"Saved      : {out}")
