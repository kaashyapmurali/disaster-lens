"""Keypoint registration example for the next-steps figure (optional; one xBD test pair; illustration only).

    uvx --with opencv-python-headless --with numpy python report/scripts/registration.py

SIFT keypoints and descriptors on the pre- and post-event crops, Lowe ratio test, RANSAC homography; run twice:
on all keypoints, and with buildings excluded (the pre-event reference footprints, dilated by a margin, are masked
out on both images) so matches come only from roads, terrain, vegetation and other non-building structure.
Writes report/bundle/images/matches{,-masked}.jpg and report/bundle/data/matches.json. OpenCV is not a project
dependency (hence uvx); run from the repository root.
"""

import json
import re

import cv2
import numpy as np

SCENE, CROP = "socal-fire_00000937", (72, 0, 952, 880)
MARGIN = 8          # px around each building footprint excluded from matching
IMAGES, DATA = "report/bundle/images", "report/bundle/data"


def load(phase: str) -> np.ndarray:
    image = cv2.imread(f"datasets/raw/test/images/{SCENE}_{phase}_disaster.png")
    x1, y1, x2, y2 = CROP
    return image[y1:y2, x1:x2]


def building_mask(shape: tuple[int, int]) -> np.ndarray:
    """255 where matching is allowed: everything except the pre-event building footprints (+ margin)."""
    with open(f"datasets/raw/test/labels/{SCENE}_pre_disaster.json") as stream:
        labels = json.load(stream)
    blocked = np.zeros(shape, np.uint8)
    for feature in labels["features"]["xy"]:
        ring = re.findall(r"([-\d.]+) ([-\d.]+)", feature["wkt"])
        points = np.array([[float(x) - CROP[0], float(y) - CROP[1]] for x, y in ring], np.int32)
        cv2.fillPoly(blocked, [points], 255)
    blocked = cv2.dilate(blocked, np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8))
    return cv2.bitwise_not(blocked)


def match(pre: np.ndarray, post: np.ndarray, mask: np.ndarray | None) -> dict:
    g1, g2 = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY), cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(g1, mask)
    k2, d2 = sift.detectAndCompute(g2, mask)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = inliers.ravel().astype(bool)
    residual = np.linalg.norm(cv2.perspectiveTransform(src[inliers][None], H)[0] - dst[inliers], axis=1)
    grid = np.stack(np.meshgrid(np.linspace(0, pre.shape[1], 23), np.linspace(0, pre.shape[0], 23)), -1).reshape(-1, 2)
    moved = np.linalg.norm(cv2.perspectiveTransform(grid.astype(np.float32)[None], H)[0] - grid, axis=1)
    return {"src": src, "dst": dst, "inliers": inliers, "stats": {
        "keypoints_pre": len(k1), "keypoints_post": len(k2), "ratio_test_matches": len(good),
        "ransac_inliers": int(inliers.sum()), "median_reprojection_px": float(np.median(residual)),
        "displacement_median_px": float(np.median(moved)), "displacement_max_px": float(moved.max())}}


def draw(pre: np.ndarray, post: np.ndarray, result: dict, path: str, mask: np.ndarray | None = None) -> None:
    size, gap = 440, 24
    s = size / pre.shape[0]
    a, b = cv2.resize(pre, (size, size)), cv2.resize(post, (size, size))
    if mask is not None:                       # darken the excluded building areas
        shade = cv2.resize(mask, (size, size)) == 0
        for image in (a, b):
            image[shade] = (image[shade] * 0.35).astype(np.uint8)
    canvas = np.full((size, 2 * size + gap, 3), 255, np.uint8)
    canvas[:, :size], canvas[:, size + gap:] = a, b
    rng = np.random.default_rng(0)
    src, dst, inliers = result["src"], result["dst"], result["inliers"]
    for indices, color, n in ((np.flatnonzero(~inliers), (80, 80, 230), 60), (np.flatnonzero(inliers), (130, 220, 60), 140)):
        for i in rng.choice(indices, min(n, len(indices)), replace=False):
            p, q = src[i] * s, dst[i] * s
            q = (int(q[0] + size + gap), int(q[1]))
            cv2.line(canvas, (int(p[0]), int(p[1])), q, color, 1, cv2.LINE_AA)
            if color[1] == 220:
                cv2.circle(canvas, (int(p[0]), int(p[1])), 2, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, q, 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    pre, post = load("pre"), load("post")
    mask = building_mask(pre.shape[:2])
    unrestricted, masked = match(pre, post, None), match(pre, post, mask)
    draw(pre, post, unrestricted, f"{IMAGES}/matches.jpg")
    draw(pre, post, masked, f"{IMAGES}/matches-masked.jpg", mask)
    out = {**unrestricted["stats"], "masked": {**masked["stats"], "margin_px": MARGIN,
                                              "excluded_share": float((mask == 0).mean())}}
    with open(f"{DATA}/matches.json", "w") as stream:
        json.dump(out, stream, indent=1)
    print(json.dumps(out, indent=1))
