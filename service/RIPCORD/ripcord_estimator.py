#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
MODEL_DIR = PROJECT_ROOT / "model"

DINO_WEIGHTS = MODEL_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
GT_PATH = PROJECT_ROOT / "service" / "RIPCORD" / "GT" / "gt.json"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

IMAGE_SIZE = 224
PATCH_GRID = 14

# ---- 开关：是否使用 DINO 作为附加判定 ----
USE_DINO = False

# ---- DINO（仅 USE_DINO=True 时生效）----
DINO_SIM_THRESHOLD = 0.85

# ---- SAM ----
MIN_SAM_SCORE = 0.30
MIN_MASK_AREA = 20

# ---- Mask 重叠判定（按实测正负样本间隙设定）----
MIN_OVERLAP_SCORE = 0.75  # coverage - EXCESS_PENALTY * excess
MIN_COVERAGE = 0.80  # SAM mask 覆盖 GT 的比例
MIN_IOU = 0.75  # 与 GT 的 IoU
MAX_AREA_RATIO = 1.8  # SAM 面积 / GT 面积 上限
EXCESS_PENALTY = 1.0  # 超出 GT 部分的扣分权重
GT_DILATE_PX = 3  # GT 容忍的位置偏差（像素），0 表示不膨胀

PRESENT_LABEL = "PRESENT"
ABSENT_LABEL = "ABSENT"

PRESENT_COLOR = (0, 255, 0)
ABSENT_COLOR = (0, 0, 255)
ROI_COLOR = (255, 0, 0)
POINT_COLOR = (255, 0, 255)
MASK_COLOR = (0, 180, 255)


class RipcordEstimator:
    def __init__(
        self,
        dino_sim_threshold: float = DINO_SIM_THRESHOLD,
        min_sam_score: float = MIN_SAM_SCORE,
        min_mask_area: int = MIN_MASK_AREA,
        use_dino: bool = USE_DINO,
    ):
        self.dino_sim_threshold = float(dino_sim_threshold)
        self.min_sam_score = float(min_sam_score)
        self.min_mask_area = int(min_mask_area)
        self.use_dino = bool(use_dino)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.gt_data = self._load_gt()
        self.roi = tuple(map(int, self.gt_data["roi"]))
        self.ripcords = self._load_gt_masks()

        self.sam2 = self._load_sam2()

        self.dino = None
        self.reference_features = None
        if self.use_dino:
            self.dino = self._load_dino()
            self.reference_features = self._build_reference_features()
        else:
            print("[DINOv3] Disabled (USE_DINO=False)")

    # ------------------------------------------------------------------
    # GT
    # ------------------------------------------------------------------
    def _load_gt(self) -> Dict[str, Any]:
        if not GT_PATH.is_file():
            raise FileNotFoundError(f"GT JSON not found: {GT_PATH}")

        data = json.loads(GT_PATH.read_text(encoding="utf-8"))

        for field in ("image", "roi", "ripcords"):
            if field not in data:
                raise RuntimeError(f"GT JSON does not contain '{field}'")

        if len(data["ripcords"]) != 4:
            raise RuntimeError("GT JSON must contain exactly 4 ripcords")

        for ripcord in data["ripcords"]:
            if "mask" not in ripcord:
                raise RuntimeError(f"Ripcord {ripcord.get('index', '?')} has no mask")

            if "file" not in ripcord["mask"]:
                raise RuntimeError(
                    f"Ripcord {ripcord.get('index', '?')} has no mask.file"
                )

        return data

    def _load_gt_masks(self) -> List[Dict[str, Any]]:
        x1, y1, x2, y2 = self.roi
        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w <= 0 or roi_h <= 0:
            raise RuntimeError(f"Invalid ROI: {self.roi}")

        results = []

        for ripcord in sorted(
            self.gt_data["ripcords"],
            key=lambda x: x["index"],
        ):
            index = int(ripcord["index"])
            center = ripcord["center"]

            mask_path = PROJECT_ROOT / ripcord["mask"]["file"]

            if not mask_path.is_file():
                raise FileNotFoundError(f"GT mask not found: {mask_path}")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise RuntimeError(f"Cannot read GT mask: {mask_path}")

            mask = mask > 127

            if mask.shape != (roi_h, roi_w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (roi_w, roi_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            area = int(np.count_nonzero(mask))

            if area < self.min_mask_area:
                raise RuntimeError(f"GT mask {index} is too small: {area}")

            # 预先计算膨胀后的 GT（容忍位置偏差）
            if GT_DILATE_PX > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * GT_DILATE_PX + 1, 2 * GT_DILATE_PX + 1),
                )
                mask_tol = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
            else:
                mask_tol = mask

            results.append(
                {
                    "index": index,
                    "center": (int(center[0]), int(center[1])),
                    "mask": mask,
                    "mask_tol": mask_tol,
                    "area": area,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _load_sam2(self) -> SAM2ImagePredictor:
        if not SAM2_CONFIG.is_file():
            raise FileNotFoundError(f"SAM2 config not found: {SAM2_CONFIG}")

        if not SAM2_CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            version_base=None,
            config_dir=str(MODEL_DIR),
        ):
            cfg = compose(
                config_name=SAM2_CONFIG.stem,
                overrides=[
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                ],
            )
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)

        _load_checkpoint(model, str(SAM2_CHECKPOINT))

        model = model.to(self.device).eval()

        print(f"[SAM2] Device: {self.device}")
        print(f"[SAM2] Config: {SAM2_CONFIG}")
        print(f"[SAM2] Checkpoint: {SAM2_CHECKPOINT}")

        return SAM2ImagePredictor(model)

    def _load_dino(self):
        if not DINO_WEIGHTS.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {DINO_WEIGHTS}")

        model = torch.hub.load(
            str(DINOV3_ROOT),
            "dinov3_vits16",
            source="local",
            weights=str(DINO_WEIGHTS),
        )

        model = model.to(self.device).eval()

        print(f"[DINOv3] Device: {self.device}")
        print(f"[DINOv3] Weights: {DINO_WEIGHTS}")

        return model

    # ------------------------------------------------------------------
    # DINO features (only used when use_dino=True)
    # ------------------------------------------------------------------
    def _dino_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        # image 为 BGR（cv2.imread 的结果）
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)

            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match "
                    f"image shape {image.shape[:2]}"
                )

            ys, xs = np.where(mask)

            if len(xs) == 0:
                raise ValueError("Mask is empty")

            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1

            image = image[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]

            background = np.zeros_like(image)
            background[mask] = image[mask]
            image = background

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            features = self.dino.forward_features(tensor)["x_norm_patchtokens"][0]

        return F.normalize(features, dim=-1)

    def _build_reference_features(self) -> torch.Tensor:
        gt_image_path = PROJECT_ROOT / self.gt_data["image"]
        gt_image = cv2.imread(str(gt_image_path))

        if gt_image is None:
            raise FileNotFoundError(f"Cannot read GT image: {gt_image_path}")

        features = []

        for ripcord in self.ripcords:
            feature = self._dino_features(
                gt_image[
                    self.roi[1] : self.roi[3],
                    self.roi[0] : self.roi[2],
                ],
                ripcord["mask"],
            )
            features.append(feature)

        return torch.stack(features, dim=0)

    # ------------------------------------------------------------------
    # SAM
    # ------------------------------------------------------------------
    def _predict_masks(
        self,
        point: Tuple[int, int],
    ) -> List[Tuple[np.ndarray, float]]:
        """返回 SAM 的全部候选 (mask, score)。"""
        point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        try:
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = self.sam2.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=None,
                            mask_input=None,
                            multimask_output=True,
                            return_logits=False,
                        )
                else:
                    masks, scores, _ = self.sam2.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=None,
                        mask_input=None,
                        multimask_output=True,
                        return_logits=False,
                    )
        except Exception as e:
            print(f"[ERROR] SAM2 prediction failed: {e}")
            return []

        masks = np.asarray(masks)
        scores = np.asarray(scores)

        if masks.ndim == 4:
            masks = masks[0]

        if scores.ndim > 1:
            scores = scores[0]

        return [(masks[i].astype(bool), float(scores[i])) for i in range(len(masks))]

    def _select_best_mask(
        self,
        candidates: List[Tuple[np.ndarray, float]],
        ripcord: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], float]:
        """在 SAM 的候选里选与 GT IoU 最高的 mask（而不是 SAM 自评分最高的）。"""
        best_mask: Optional[np.ndarray] = None
        best_score = 0.0
        best_iou = -1.0

        for mask, score in candidates:
            if int(np.count_nonzero(mask)) < self.min_mask_area:
                continue

            iou = self._mask_overlap(
                mask,
                ripcord["mask"],
                ripcord["mask_tol"],
            )["iou"]

            if iou > best_iou:
                best_mask, best_score, best_iou = mask, score, iou

        return best_mask, best_score

    # ------------------------------------------------------------------
    # Mask overlap
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_overlap(
        mask: np.ndarray,
        gt_mask: np.ndarray,
        gt_mask_tol: np.ndarray,
    ) -> Dict[str, float]:
        gt_area = max(int(np.count_nonzero(gt_mask)), 1)
        mask_area = int(np.count_nonzero(mask))

        inter = int(np.count_nonzero(mask & gt_mask))
        outside = int(np.count_nonzero(mask & ~gt_mask_tol))
        union = max(gt_area + mask_area - inter, 1)

        coverage = inter / gt_area
        excess = outside / gt_area
        area_ratio = mask_area / gt_area
        iou = inter / union

        overlap_score = coverage - EXCESS_PENALTY * excess

        return {
            "coverage": float(coverage),
            "excess": float(excess),
            "area_ratio": float(area_ratio),
            "iou": float(iou),
            "overlap_score": float(overlap_score),
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_mask(
        self,
        roi_image: np.ndarray,
        mask: Optional[np.ndarray],
        sam_score: float,
        ripcord: Dict[str, Any],
        ref_idx: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": ABSENT_LABEL,
            "present": False,
            "sam_score": float(sam_score),
            "mask_area": 0,
            "coverage": 0.0,
            "excess": 0.0,
            "area_ratio": 0.0,
            "iou": 0.0,
            "overlap_score": 0.0,
            "overlap_ok": False,
            # DINO 关闭时 dino_similarity 恒为 0.0（保持 float，避免下游格式化报错），
            # 用 dino_enabled 区分"未启用"和"相似度为 0"
            "dino_enabled": self.use_dino,
            "dino_similarity": 0.0,
            "dino_ok": False,
        }

        if mask is None:
            return result

        area = int(np.count_nonzero(mask))
        result["mask_area"] = area

        if area < self.min_mask_area:
            return result

        # ---- Mask 重叠判定 ----
        overlap = self._mask_overlap(
            mask,
            ripcord["mask"],
            ripcord["mask_tol"],
        )
        result.update(overlap)

        overlap_ok = (
            overlap["overlap_score"] >= MIN_OVERLAP_SCORE
            and overlap["coverage"] >= MIN_COVERAGE
            and overlap["iou"] >= MIN_IOU
            and overlap["area_ratio"] <= MAX_AREA_RATIO
            and sam_score >= self.min_sam_score
        )
        result["overlap_ok"] = overlap_ok

        present = overlap_ok

        # ---- DINO 附加判定（可选）----
        if self.use_dino:
            candidate_feature = self._dino_features(roi_image, mask)  # [196, C]
            ref = self.reference_features[ref_idx]  # [196, C]
            dino_sim = float((candidate_feature * ref).sum(dim=-1).mean().item())
            dino_ok = dino_sim >= self.dino_sim_threshold

            result["dino_similarity"] = dino_sim
            result["dino_ok"] = dino_ok

            present = present and dino_ok

        result["present"] = present
        result["label"] = PRESENT_LABEL if present else ABSENT_LABEL

        return result

    # ------------------------------------------------------------------
    # Debug drawing
    # ------------------------------------------------------------------
    def _draw_debug(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
    ) -> np.ndarray:
        debug = image.copy()
        x1, y1, x2, y2 = self.roi

        cv2.rectangle(debug, (x1, y1), (x2, y2), ROI_COLOR, 3)

        for result in results:
            index = result["index"]
            gx, gy = result["center"]
            evaluation = result["evaluation"]
            mask = result["mask"]

            present = evaluation["present"]
            color = PRESENT_COLOR if present else ABSENT_COLOR

            if present and mask is not None:
                full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask.astype(np.uint8) * 255

                overlay = debug.copy()
                overlay[full_mask > 0] = MASK_COLOR
                debug = cv2.addWeighted(overlay, 0.25, debug, 0.75, 0)

                contours, _ = cv2.findContours(
                    full_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(debug, contours, -1, color, 3)

            cv2.circle(debug, (gx, gy), 9, POINT_COLOR, 2)

            tx = gx + 18
            ty = gy - 18

            cv2.putText(
                debug,
                f"Ripcord {index}: {evaluation['label']}",
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )

            line2 = (
                f"COV={evaluation['coverage']:.2f} "
                f"OV={evaluation['overlap_score']:.2f} "
                f"IoU={evaluation['iou']:.2f} "
                f"AR={evaluation['area_ratio']:.1f}"
            )
            if evaluation["dino_enabled"]:
                line2 += f" DINO={evaluation['dino_similarity']:.2f}"

            cv2.putText(
                debug,
                line2,
                (tx, ty + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        title = "SAM2 + DINOv3 RIPCORD" if self.use_dino else "SAM2 MASK RIPCORD"
        cv2.rectangle(debug, (10, 10), (500, 70), (0, 0, 0), -1)
        cv2.putText(
            debug,
            title,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return debug

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        # 注意：参数名为 rgb，但实际按 cv2.imread 的 BGR 图像处理
        if rgb is None or not isinstance(rgb, np.ndarray):
            raise ValueError("rgb must be a numpy.ndarray")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")

        x1, y1, x2, y2 = self.roi
        h, w = rgb.shape[:2]

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(f"ROI {self.roi} is outside input image size {w}x{h}")

        roi_image = rgb[y1:y2, x1:x2]

        if roi_image.size == 0:
            raise RuntimeError("Input ROI is empty")

        roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.sam2.set_image(roi_rgb)
            else:
                self.sam2.set_image(roi_rgb)

        results = []

        for i, ripcord in enumerate(self.ripcords):
            index = ripcord["index"]
            gx, gy = ripcord["center"]

            rx = gx - x1
            ry = gy - y1

            if rx < 0 or ry < 0 or rx >= roi_image.shape[1] or ry >= roi_image.shape[0]:
                mask = None
                sam_score = 0.0
            else:
                candidates = self._predict_masks((rx, ry))
                mask, sam_score = self._select_best_mask(candidates, ripcord)

            evaluation = self._classify_mask(
                roi_image,
                mask,
                sam_score,
                ripcord=ripcord,
                ref_idx=i,
            )

            results.append(
                {
                    "index": index,
                    "center": ripcord["center"],
                    "mask": mask,
                    "evaluation": evaluation,
                }
            )

        debug_image = self._draw_debug(rgb, results)

        present_indices = [r["index"] for r in results if r["evaluation"]["present"]]
        absent_indices = [r["index"] for r in results if not r["evaluation"]["present"]]

        detected_index = present_indices[0] if present_indices else None

        output = {
            "index": detected_index,
            "detected_index": detected_index,
            "present_indices": present_indices,
            "absent_indices": absent_indices,
            "ripcords": {},
            "debug_image": debug_image,
        }

        for result in results:
            output["ripcords"][result["index"]] = dict(result["evaluation"])

        return output


if __name__ == "__main__":
    import sys

    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "dino_test" / "hard.png"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    estimator = RipcordEstimator()
    result = estimator.predict(image)

    print(f"Device: {estimator.device}")
    print(f"Use DINO: {estimator.use_dino}")
    if estimator.use_dino:
        print(f"DINO threshold: {estimator.dino_sim_threshold}")

    for index in sorted(result["ripcords"].keys()):
        item = result["ripcords"][index]

        line = (
            f"Ripcord {index}: {item['label']} | "
            f"SAM={item['sam_score']:.4f} | "
            f"cov={item['coverage']:.3f} | "
            f"excess={item['excess']:.3f} | "
            f"AR={item['area_ratio']:.2f} | "
            f"IoU={item['iou']:.3f} | "
            f"OV={item['overlap_score']:.3f}"
        )
        if item["dino_enabled"]:
            line += f" | DINO={item['dino_similarity']:.4f}"
        print(line)

    print(f"PRESENT: {result['present_indices']}")
    print(f"ABSENT:  {result['absent_indices']}")
    print(f"FINAL INDEX: {result['detected_index']}")

    output_path = PROJECT_ROOT / "dino_test" / "ripcord_debug.png"
    cv2.imwrite(str(output_path), result["debug_image"])

    print(f"Debug image: {output_path}")
