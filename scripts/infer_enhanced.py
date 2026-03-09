#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本推理可视化脚本 (Fixed & Robust)
"""

import os, sys, argparse, yaml
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import model
try:
    from enhanced_detector import EnhancedAVTopDetector
except ImportError:
    # 如果脚本在 scripts/ 目录下运行，可能需要调整导入路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_detector import EnhancedAVTopDetector

from dataset import AVFromCSV


class InferenceVisualizer:
    def __init__(self, model, class_names, device, output_dir):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        (self.output_dir / 'frames').mkdir(exist_ok=True)
        (self.output_dir / 'analysis').mkdir(exist_ok=True)

    @torch.no_grad()
    def visualize_sample(self, video, audio, label=None, sample_name="sample"):
        self.model.eval()

        # Batch维度处理
        if video.dim() == 4: video = video.unsqueeze(0)
        if audio.dim() == 3: audio = audio.unsqueeze(0)

        video = video.to(self.device)
        audio = audio.to(self.device)

        # 前向
        outputs = self.model(video, audio, return_aux=True)

        # 1. 解析预测结果
        # 适配返回结构：字典 或 Tensor
        if isinstance(outputs, dict):
            logits = outputs.get('clip_logits', outputs.get('logits'))
            if logits is None:  # Fallback for simple dict
                # 假设第一个 value 是 logits
                logits = list(outputs.values())[0]
        else:
            logits = outputs

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = logits.argmax(dim=1).item()

        print(f"Sample: {sample_name}")
        print(f"Prediction: {self.class_names[pred_idx]} ({probs[pred_idx]:.2%})")
        if label is not None:
            print(f"Ground Truth: {self.class_names[label]}")

        # 2. 生成综合分析图
        self._create_dashboard(video, audio, outputs, probs, pred_idx, label, sample_name)

    def _create_dashboard(self, video, audio, outputs, probs, pred, label, name):
        """生成一张包含输入、特征、注意力、预测的综合仪表盘"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig)

        # --- Row 1: Input & CAVA ---
        # 1. Video Middle Frame (FIXED)
        ax_vid = fig.add_subplot(gs[0, 0])
        vid_np = video[0].cpu().numpy()  # [T, 3, H, W]
        mid_idx = len(vid_np) // 2
        mid_frame = vid_np[mid_idx].transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        # 简单归一化处理，防止数据显示异常
        if mid_frame.max() > 1.0:
            mid_frame = mid_frame / 255.0
        mid_frame = np.clip(mid_frame, 0, 1)

        ax_vid.imshow(mid_frame)
        ax_vid.set_title("Input Video (Mid Frame)")
        ax_vid.axis('off')

        # 2. Audio Spectrogram
        ax_aud = fig.add_subplot(gs[0, 1])
        aud_np = audio[0].cpu().numpy()  # [T, Mel, F]
        # Concat time frames for visualization
        aud_viz = np.concatenate([aud_np[t] for t in range(aud_np.shape[0])], axis=1)
        ax_aud.imshow(aud_viz, origin='lower', aspect='auto', cmap='magma')
        ax_aud.set_title("Input Audio Log-Mel")
        ax_aud.axis('off')

        # 3. CAVA Gate (If available)
        ax_gate = fig.add_subplot(gs[0, 2:])
        if isinstance(outputs, dict) and 'causal_gate' in outputs and outputs['causal_gate'] is not None:
            gate = outputs['causal_gate'][0].cpu().numpy()
            if gate.ndim > 1: gate = gate.mean(axis=1)  # Handle [T, D]
            ax_gate.plot(gate, 'o-', color='purple', lw=2)
            ax_gate.set_ylim(0, 1.1)
            ax_gate.set_title("CAVA Alignment Gate")
            ax_gate.set_xlabel("Time Step")
            ax_gate.set_ylabel("Gate Value")
            ax_gate.grid(True, alpha=0.3)
        else:
            ax_gate.text(0.5, 0.5, "CAVA Gate Not Available", ha='center')

        # --- Row 2: Features & Attention ---
        # 4. Feature Similarity Matrix (Video vs Audio)
        ax_sim = fig.add_subplot(gs[1, 0:2])
        if isinstance(outputs, dict):
            # 优先用投影后的特征
            v_feat = outputs.get('video_proj', outputs.get('video_emb'))
            a_feat = outputs.get('audio_aligned', outputs.get('audio_emb'))

            if v_feat is not None and a_feat is not None:
                v = v_feat[0].cpu().numpy()  # [T, D]
                a = a_feat[0].cpu().numpy()  # [T, D]
                # Normalize
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
                sim_mat = v @ a.T
                im = ax_sim.imshow(sim_mat, cmap='viridis', origin='lower')
                ax_sim.set_title("Video-Audio Feature Similarity")
                ax_sim.set_xlabel("Audio Time")
                ax_sim.set_ylabel("Video Time")
                plt.colorbar(im, ax=ax_sim)
            else:
                ax_sim.text(0.5, 0.5, "Features Not Available", ha='center')

        # 5. Attention / Weights
        ax_attn = fig.add_subplot(gs[1, 2:])
        if isinstance(outputs, dict):
            # Check for MIL weights first
            weights = outputs.get('weights')  # [B, T]
            if weights is not None:
                w = weights[0].cpu().numpy()
                ax_attn.bar(range(len(w)), w, color='orange', alpha=0.7)
                ax_attn.set_title("MIL Temporal Attention Weights")
                ax_attn.set_xlabel("Time Step")
            else:
                ax_attn.text(0.5, 0.5, "Attention Weights Not Available", ha='center')

        # --- Row 3: Predictions ---
        # 6. Top-5 Probabilities
        ax_prob = fig.add_subplot(gs[2, 0:2])
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_probs = probs[top5_idx]
        top5_names = [self.class_names[i] for i in top5_idx]

        colors = ['green' if i == pred else 'gray' for i in top5_idx]
        ax_prob.barh(top5_names, top5_probs, color=colors)
        ax_prob.set_xlim(0, 1)
        ax_prob.invert_yaxis()
        ax_prob.set_title(f"Prediction: {self.class_names[pred]}")

        # 7. Frame-level Predictions (If available)
        ax_frame = fig.add_subplot(gs[2, 2:])
        if isinstance(outputs, dict) and 'seg_logits' in outputs:
            # [B, T, C]
            seg_logits = outputs['seg_logits'][0]
            seg_probs = F.softmax(seg_logits, dim=1).cpu().numpy()  # [T, C]

            # Plot probability of the predicted class over time
            ax_frame.plot(seg_probs[:, pred], 'o-', label=f'Class: {self.class_names[pred]}', color='blue')
            if label is not None and label != pred:
                ax_frame.plot(seg_probs[:, label], 'x--', label=f'True: {self.class_names[label]}', color='green')

            ax_frame.set_ylim(0, 1.1)
            ax_frame.set_title("Frame-level Confidence (Seg Logits)")
            ax_frame.set_xlabel("Time Step")
            ax_frame.legend()
            ax_frame.grid(True, alpha=0.3)
        else:
            ax_frame.text(0.5, 0.5, "Frame Predictions Not Available", ha='center')

        plt.tight_layout()
        save_path = self.output_dir / 'analysis' / f'{name}_dashboard.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved dashboard to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='./inference_results')
    parser.add_argument('--dataset', type=str, help='CSV dataset for sampling')
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()

    # Load Config (FIXED ENCODING for Windows)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model_cfg = cfg.get("model", {})
    model_cfg["num_classes"] = cfg["data"]["num_classes"]

    # Init Model
    model = EnhancedAVTopDetector({
        "model": model_cfg,
        "fusion": model_cfg.get("fusion", {}),
        "cava": cfg.get("cava", {})
    }).to(device)

    # Load Weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    # Load Data
    if args.dataset:
        dataset = AVFromCSV(
            args.dataset,
            cfg["data"].get("data_root"),
            cfg["data"]["num_classes"],
            cfg["data"]["class_names"],
            cfg.get("video", {}),
            cfg.get("audio", {}),
            is_unlabeled=False
        )
        # Handle tuple return from dataset
        data_item = dataset[args.sample_idx]
        video = data_item[0]
        audio = data_item[1]
        label = data_item[2]

        sample_name = f"sample_{args.sample_idx}"
    else:
        print("Please provide --dataset to load a sample.")
        return

    # Run
    viz = InferenceVisualizer(model, cfg["data"]["class_names"], device, args.output)
    viz.visualize_sample(video, audio, label.item() if torch.is_tensor(label) else label, sample_name)


if __name__ == "__main__":
    main()