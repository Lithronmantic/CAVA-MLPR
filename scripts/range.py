import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
import cv2
import librosa
import os
import pandas as pd
from matplotlib.gridspec import GridSpec


class AnomalyDiagnoser:
    """音视频异常样本诊断工具"""

    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

    def load_sample(self, filename):
        """加载音视频样本"""
        # 查找文件
        video_path = None
        for root, dirs, files in os.walk(self.dataset_root):
            if filename in files:
                video_path = os.path.join(root, filename)
                break

        if video_path is None:
            raise FileNotFoundError(f"找不到文件: {filename}")

        # 音频路径
        base_path = os.path.splitext(video_path)[0]
        audio_path = None
        for ext in ['.flac', '.wav']:
            a_path = base_path + ext
            if os.path.exists(a_path):
                audio_path = a_path
                break

        if audio_path is None:
            raise FileNotFoundError(f"找不到音频文件: {filename}")

        return video_path, audio_path

    def extract_detailed_signals(self, video_path, audio_path):
        """提取详细的音视频信号特征"""

        # ==================== 视频分析 ====================
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        visual_signals = {
            'mean_intensity': [],  # 平均亮度
            'std_intensity': [],  # 亮度标准差（对比度）
            'edge_density': [],  # 边缘密度
            'motion_energy': []  # 运动能量
        }

        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 亮度特征
            visual_signals['mean_intensity'].append(np.mean(gray))
            visual_signals['std_intensity'].append(np.std(gray))

            # 边缘特征（使用Canny）
            edges = cv2.Canny(gray, 50, 150)
            visual_signals['edge_density'].append(np.sum(edges > 0) / edges.size)

            # 运动特征（帧间差分）
            if prev_frame is not None:
                motion = np.abs(gray.astype(float) - prev_frame.astype(float))
                visual_signals['motion_energy'].append(np.mean(motion))
            else:
                visual_signals['motion_energy'].append(0)

            prev_frame = gray
            frame_idx += 1

        cap.release()

        # 转为numpy数组
        for key in visual_signals:
            visual_signals[key] = np.array(visual_signals[key])

        # 视频质量指标
        video_quality = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0,
            'mean_brightness': np.mean(visual_signals['mean_intensity']),
            'brightness_std': np.std(visual_signals['mean_intensity']),
            'has_frozen_frames': self._detect_frozen_frames(visual_signals['motion_energy']),
            'has_black_frames': self._detect_black_frames(visual_signals['mean_intensity']),
            'brightness_variance': np.var(visual_signals['mean_intensity'])
        }

        # ==================== 音频分析 ====================
        y, sr = librosa.load(audio_path, sr=None)

        # 计算与视频帧对应的音频特征
        samples_per_frame = int(len(y) / frame_count) if frame_count > 0 else 0

        if samples_per_frame > 0:
            audio_rms = librosa.feature.rms(
                y=y,
                frame_length=2048,
                hop_length=samples_per_frame,
                center=True
            )[0]

            audio_zcr = librosa.feature.zero_crossing_rate(
                y=y,
                frame_length=2048,
                hop_length=samples_per_frame
            )[0]

            # 对齐长度
            min_len = min(len(audio_rms), frame_count)
            audio_rms = audio_rms[:min_len]
            audio_zcr = audio_zcr[:min_len]
        else:
            audio_rms = np.zeros(frame_count)
            audio_zcr = np.zeros(frame_count)

        audio_signals = {
            'rms': audio_rms,
            'zcr': audio_zcr
        }

        # 音频质量指标
        audio_quality = {
            'sample_rate': sr,
            'duration': len(y) / sr,
            'total_samples': len(y),
            'mean_rms': np.mean(audio_rms),
            'rms_std': np.std(audio_rms),
            'has_silence': self._detect_silence(audio_rms),
            'has_clipping': self._detect_clipping(y),
            'snr_estimate': self._estimate_snr(audio_rms)
        }

        return visual_signals, audio_signals, video_quality, audio_quality

    def _detect_frozen_frames(self, motion_energy, threshold=0.5):
        """检测静止帧（视频卡顿）"""
        frozen_ratio = np.sum(motion_energy < threshold) / len(motion_energy)
        return frozen_ratio > 0.3  # 超过30%的帧静止

    def _detect_black_frames(self, mean_intensity, threshold=10):
        """检测黑屏帧"""
        black_ratio = np.sum(mean_intensity < threshold) / len(mean_intensity)
        return black_ratio > 0.1  # 超过10%的帧是黑屏

    def _detect_silence(self, rms, threshold=0.0005):  # 更严格的静音阈值
        """检测静音段 - 只有极低的RMS才算静音"""
        silence_ratio = np.sum(rms < threshold) / len(rms)
        return silence_ratio > 0.8  # 超过80%才认为是大量静音

    def _detect_clipping(self, audio_signal, threshold=0.995):  # 更严格的削波阈值
        """检测音频削波失真"""
        clip_ratio = np.sum(np.abs(audio_signal) > threshold) / len(audio_signal)
        return clip_ratio > 0.05  # 超过5%才认为有明显削波

    def _estimate_snr(self, rms):
        """估算信噪比"""
        if len(rms) == 0:
            return 0
        signal_power = np.mean(rms ** 2)
        noise_power = np.percentile(rms ** 2, 10)  # 底部10%作为噪声
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return snr_db
        return 0

    def _calculate_peak_clarity(self, correlation, peak_idx):
        """正确计算峰值清晰度：峰值突出程度"""
        peak_value = correlation[peak_idx]

        # 方法1：峰值与均值的标准化距离
        mean_corr = np.mean(correlation)
        std_corr = np.std(correlation)

        if std_corr > 1e-6:
            clarity_1 = (peak_value - mean_corr) / std_corr
        else:
            clarity_1 = 0

        # 方法2：峰值与次高峰的比值
        # 找到次高峰（距离主峰至少50帧）
        window = 50
        local_max_indices = signal.find_peaks(correlation, distance=window)[0]

        if len(local_max_indices) >= 2:
            sorted_peaks = sorted([(correlation[i], i) for i in local_max_indices], reverse=True)
            second_peak_value = sorted_peaks[1][0] if len(sorted_peaks) > 1 else mean_corr
            clarity_2 = peak_value / (second_peak_value + 1e-6) if second_peak_value > 0 else clarity_1
        else:
            clarity_2 = clarity_1

        # 返回更保守的值
        return max(clarity_1, clarity_2)

    def calculate_correlation(self, visual_signal, audio_signal):
        """计算互相关并返回详细信息"""
        # 归一化
        if np.std(visual_signal) > 1e-5:
            v_norm = (visual_signal - np.mean(visual_signal)) / np.std(visual_signal)
        else:
            v_norm = visual_signal - np.mean(visual_signal)

        if np.std(audio_signal) > 1e-5:
            a_norm = (audio_signal - np.mean(audio_signal)) / np.std(audio_signal)
        else:
            a_norm = audio_signal - np.mean(audio_signal)

        # 互相关
        correlation = signal.correlate(v_norm, a_norm, mode='full')
        lags = signal.correlation_lags(len(v_norm), len(a_norm), mode='full')

        # 找到峰值
        peak_idx = np.argmax(correlation)
        lag_frames = lags[peak_idx]
        peak_corr = correlation[peak_idx]

        # 归一化相关系数
        norm_corr = peak_corr / len(v_norm)

        # 找到次优峰值（判断是否有多个对齐候选）
        peak_indices = signal.find_peaks(correlation, height=peak_corr * 0.7, distance=50)[0]

        return {
            'lag': lag_frames,
            'peak_correlation': norm_corr,
            'correlation_full': correlation,
            'lags_full': lags,
            'num_candidate_peaks': len(peak_indices),  # 候选峰值数量
            'peak_clarity': self._calculate_peak_clarity(correlation, peak_idx)
        }

    def diagnose_sample(self, filename, save_dir='diagnosis_results'):
        """诊断单个样本并生成报告"""

        print(f"\n{'=' * 60}")
        print(f"🔍 正在诊断样本: {filename}")
        print(f"{'=' * 60}")

        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 加载数据
            video_path, audio_path = self.load_sample(filename)
            print(f"✅ 视频路径: {video_path}")
            print(f"✅ 音频路径: {audio_path}")

            # 提取信号
            print("\n📊 正在提取信号特征...")
            v_signals, a_signals, v_quality, a_quality = self.extract_detailed_signals(
                video_path, audio_path
            )

            # 对齐长度
            min_len = min(len(v_signals['mean_intensity']), len(a_signals['rms']))
            v_energy = v_signals['mean_intensity'][:min_len]
            a_energy = a_signals['rms'][:min_len]

            # 计算互相关
            print("📊 正在计算互相关...")
            corr_info = self.calculate_correlation(v_energy, a_energy)

            # 生成诊断报告
            diagnosis = self._generate_diagnosis(
                filename, v_signals, a_signals, v_quality, a_quality, corr_info
            )

            # 打印报告
            self._print_diagnosis(diagnosis)

            # 保存可视化
            self._visualize_diagnosis(
                filename, v_signals, a_signals, v_quality, a_quality, corr_info,
                save_dir
            )

            # 保存文本报告
            self._save_text_report(filename, diagnosis, save_dir)

            return diagnosis

        except Exception as e:
            print(f"❌ 诊断失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_diagnosis(self, filename, v_signals, a_signals, v_quality, a_quality, corr_info):
        """生成诊断结果"""

        issues = []
        anomaly_source = []
        confidence = "低"

        # 视频问题检测
        if v_quality['has_black_frames']:
            issues.append("检测到黑屏帧（>10%的帧亮度<10）")
            anomaly_source.append("视频")

        if v_quality['has_frozen_frames']:
            issues.append("检测到大量静止帧（>30%的帧运动能量<0.5）")
            anomaly_source.append("视频")

        if v_quality['brightness_variance'] < 1.0:
            issues.append("视频亮度几乎不变（方差<1.0，可能是静态画面）")
            anomaly_source.append("视频")

        # 音频问题检测 - 更严格的标准
        if a_quality['mean_rms'] < 0.001:  # 极低的响度
            issues.append(f"音频响度极低（平均RMS={a_quality['mean_rms']:.6f}，可能是静音）")
            anomaly_source.append("音频")

        if a_quality['has_clipping']:
            issues.append("检测到音频削波失真（>1%样本被削波）")
            anomaly_source.append("音频")

        if a_quality['snr_estimate'] < 5:  # 更严格的SNR阈值
            issues.append(f"音频信噪比极低（SNR≈{a_quality['snr_estimate']:.1f}dB）")
            anomaly_source.append("音频")

        # 时长不匹配 - 提高阈值
        duration_diff = abs(v_quality['duration'] - a_quality['duration'])
        if duration_diff > 5.0:  # 超过5秒才认为异常
            issues.append(f"音视频时长严重不匹配（差异: {duration_diff:.2f}秒）")
            anomaly_source.append("同步")

        # 互相关质量 - 修正后的标准
        if corr_info['peak_correlation'] < 0.2:  # 相关系数太低
            issues.append(f"音视频相关性极低（相关系数={corr_info['peak_correlation']:.3f}）")
            anomaly_source.append("信号相关性")

        if corr_info['peak_clarity'] < 1.5:  # 峰值不突出
            issues.append(f"互相关峰值不突出（清晰度={corr_info['peak_clarity']:.2f}，存在多个候选对齐）")
            anomaly_source.append("对齐不确定")

        # 极端延迟
        if abs(corr_info['lag']) > 100:
            issues.append(f"延迟超出合理范围（{corr_info['lag']}帧 ≈ {corr_info['lag'] / 30:.1f}秒）")
            anomaly_source.append("极端延迟")

        # 判断置信度
        if len(issues) == 0:
            confidence = "正常"
            issues.append("✅ 未检测到明显异常，音视频对齐质量良好")
        elif len(issues) >= 3:
            confidence = "高"
        elif len(issues) >= 1:
            confidence = "中"

        # 主要异常来源
        if anomaly_source:
            primary_source = max(set(anomaly_source), key=anomaly_source.count)
        else:
            primary_source = "无异常"

        return {
            'filename': filename,
            'lag': corr_info['lag'],
            'issues': issues,
            'primary_source': primary_source,
            'confidence': confidence,
            'v_quality': v_quality,
            'a_quality': a_quality,
            'corr_info': corr_info
        }

    def _print_diagnosis(self, diagnosis):
        """打印诊断结果"""
        print(f"\n{'=' * 60}")
        print("📋 诊断报告")
        print(f"{'=' * 60}")
        print(f"文件名: {diagnosis['filename']}")
        print(f"延迟: {diagnosis['lag']} 帧 ({diagnosis['lag'] / 30:.2f} 秒)")

        # 根据置信度使用不同的显示
        if diagnosis['confidence'] == "正常":
            print(f"诊断结果: ✅ {diagnosis['confidence']} - 音视频对齐良好")
        else:
            print(f"异常置信度: ⚠️ {diagnosis['confidence']}")
            print(f"主要异常来源: {diagnosis['primary_source']}")

        print(f"\n🔍 诊断详情:")
        for i, issue in enumerate(diagnosis['issues'], 1):
            print(f"  {i}. {issue}")

        print(f"\n📹 视频质量指标:")
        print(f"  - 帧率: {diagnosis['v_quality']['fps']:.2f} fps")
        print(f"  - 时长: {diagnosis['v_quality']['duration']:.2f} 秒")
        print(f"  - 平均亮度: {diagnosis['v_quality']['mean_brightness']:.2f}")
        print(f"  - 亮度标准差: {diagnosis['v_quality']['brightness_std']:.2f}")
        print(f"  - 静止帧: {'是' if diagnosis['v_quality']['has_frozen_frames'] else '否'}")
        print(f"  - 黑屏帧: {'是' if diagnosis['v_quality']['has_black_frames'] else '否'}")

        print(f"\n🔊 音频质量指标:")
        print(f"  - 采样率: {diagnosis['a_quality']['sample_rate']} Hz")
        print(f"  - 时长: {diagnosis['a_quality']['duration']:.2f} 秒")
        print(f"  - 平均响度(RMS): {diagnosis['a_quality']['mean_rms']:.6f}")
        print(f"  - 信噪比估计: {diagnosis['a_quality']['snr_estimate']:.1f} dB")
        print(f"  - 削波失真: {'是' if diagnosis['a_quality']['has_clipping'] else '否'}")

        print(f"\n🔗 互相关分析:")
        print(f"  - 峰值相关系数: {diagnosis['corr_info']['peak_correlation']:.4f}")
        print(f"  - 峰值清晰度: {diagnosis['corr_info']['peak_clarity']:.2f}")

        # 给出质量评级
        if diagnosis['corr_info']['peak_correlation'] > 0.5:
            corr_quality = "优秀"
        elif diagnosis['corr_info']['peak_correlation'] > 0.3:
            corr_quality = "良好"
        elif diagnosis['corr_info']['peak_correlation'] > 0.2:
            corr_quality = "一般"
        else:
            corr_quality = "较差"

        print(f"  - 对齐质量评级: {corr_quality}")
        print(f"{'=' * 60}\n")

    def _visualize_diagnosis(self, filename, v_signals, a_signals, v_quality, a_quality, corr_info, save_dir):
        """生成可视化诊断图"""

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 对齐长度
        min_len = min(len(v_signals['mean_intensity']), len(a_signals['rms']))
        time_axis = np.arange(min_len) / 30  # 假设30fps

        # 1. 视频信号 - 亮度
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_axis, v_signals['mean_intensity'][:min_len], 'b-', linewidth=1)
        ax1.set_title('Video Signal - Mean Intensity', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Brightness')
        ax1.grid(alpha=0.3)

        # 2. 视频信号 - 运动能量
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_axis, v_signals['motion_energy'][:min_len], 'g-', linewidth=1)
        ax2.set_title('Video Signal - Motion Energy', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Motion Magnitude')
        ax2.grid(alpha=0.3)

        # 3. 音频信号 - RMS能量
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_axis, a_signals['rms'][:min_len], 'r-', linewidth=1)
        ax3.set_title('Audio Signal - RMS Energy', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('RMS Amplitude')
        ax3.grid(alpha=0.3)

        # 4. 音频信号 - 过零率
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_axis, a_signals['zcr'][:min_len], 'm-', linewidth=1)
        ax4.set_title('Audio Signal - Zero Crossing Rate', fontweight='bold')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('ZCR')
        ax4.grid(alpha=0.3)

        # 5. 归一化后的音视频对比
        ax5 = fig.add_subplot(gs[2, :])
        v_norm = (v_signals['mean_intensity'][:min_len] - np.mean(v_signals['mean_intensity'][:min_len])) / (
                    np.std(v_signals['mean_intensity'][:min_len]) + 1e-6)
        a_norm = (a_signals['rms'][:min_len] - np.mean(a_signals['rms'][:min_len])) / (
                    np.std(a_signals['rms'][:min_len]) + 1e-6)

        ax5.plot(time_axis, v_norm, 'b-', linewidth=1.5, alpha=0.7, label='Video (Normalized)')
        ax5.plot(time_axis, a_norm, 'r-', linewidth=1.5, alpha=0.7, label='Audio (Normalized)')
        ax5.set_title(f'Normalized Signals Comparison (Detected Lag: {corr_info["lag"]} frames)', fontweight='bold')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Normalized Amplitude')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. 互相关结果
        ax6 = fig.add_subplot(gs[3, :])
        ax6.plot(corr_info['lags_full'], corr_info['correlation_full'], 'k-', linewidth=1)
        ax6.axvline(corr_info['lag'], color='red', linestyle='--', linewidth=2,
                    label=f'Peak at {corr_info["lag"]} frames')
        ax6.axvline(0, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Zero lag')
        ax6.set_title(f'Cross-Correlation (Peak Clarity: {corr_info["peak_clarity"]:.2f})', fontweight='bold')
        ax6.set_xlabel('Lag (frames)')
        ax6.set_ylabel('Correlation')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 设置整体标题
        fig.suptitle(f'Anomaly Diagnosis: {filename}', fontsize=16, fontweight='bold', y=0.995)

        # 保存
        safe_filename = filename.replace('.avi', '').replace('.mp4', '')
        save_path = os.path.join(save_dir, f'{safe_filename}_diagnosis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 可视化已保存: {save_path}")

    def _save_text_report(self, filename, diagnosis, save_dir):
        """保存文本报告"""
        safe_filename = filename.replace('.avi', '').replace('.mp4', '')
        report_path = os.path.join(save_dir, f'{safe_filename}_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"音视频异常诊断报告\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"文件名: {diagnosis['filename']}\n")
            f.write(f"延迟: {diagnosis['lag']} 帧 ({diagnosis['lag'] / 30:.2f} 秒)\n")

            if diagnosis['confidence'] == "正常":
                f.write(f"诊断结果: ✅ {diagnosis['confidence']}\n\n")
            else:
                f.write(f"异常置信度: {diagnosis['confidence']}\n")
                f.write(f"主要异常来源: {diagnosis['primary_source']}\n\n")

            f.write(f"诊断详情:\n")
            for i, issue in enumerate(diagnosis['issues'], 1):
                f.write(f"  {i}. {issue}\n")

            f.write(f"\n视频质量指标:\n")
            for key, value in diagnosis['v_quality'].items():
                f.write(f"  - {key}: {value}\n")

            f.write(f"\n音频质量指标:\n")
            for key, value in diagnosis['a_quality'].items():
                f.write(f"  - {key}: {value}\n")

            f.write(f"\n互相关分析:\n")
            f.write(f"  - 峰值相关系数: {diagnosis['corr_info']['peak_correlation']:.4f}\n")
            f.write(f"  - 峰值清晰度: {diagnosis['corr_info']['peak_clarity']:.2f}\n")
            f.write(f"  - 候选峰值数量: {diagnosis['corr_info']['num_candidate_peaks']}\n")

        print(f"📄 文本报告已保存: {report_path}")

    def batch_diagnose(self, anomaly_csv_path, top_n=10, save_dir='diagnosis_results'):
        """批量诊断异常样本"""

        print(f"\n{'=' * 60}")
        print(f"🚀 开始批量诊断 (Top {top_n} 异常样本)")
        print(f"{'=' * 60}\n")

        # 读取异常文件列表
        df = pd.read_csv(anomaly_csv_path, encoding='utf-8-sig')

        # 按延迟绝对值排序，取top_n
        df['abs_delay'] = df['延迟(帧)'].abs()
        top_anomalies = df.nlargest(top_n, 'abs_delay')

        results = []

        for idx, row in top_anomalies.iterrows():
            filename = row['文件名']
            diagnosis = self.diagnose_sample(filename, save_dir)
            if diagnosis:
                results.append(diagnosis)

        # 生成汇总报告
        self._generate_summary_report(results, save_dir)

        print(f"\n✅ 批量诊断完成！结果保存在: {save_dir}")

        return results

    def _generate_summary_report(self, results, save_dir):
        """生成汇总报告"""

        summary_path = os.path.join(save_dir, 'summary_report.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"批量诊断汇总报告\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"总诊断样本数: {len(results)}\n\n")

            # 统计异常来源
            sources = [r['primary_source'] for r in results]
            source_counts = pd.Series(sources).value_counts()

            f.write(f"异常来源统计:\n")
            for source, count in source_counts.items():
                f.write(f"  - {source}: {count} 个样本\n")

            f.write(f"\n详细列表:\n")
            f.write(f"{'-' * 60}\n")

            for r in results:
                f.write(f"\n文件: {r['filename']}\n")
                f.write(f"  延迟: {r['lag']} 帧\n")
                f.write(f"  异常来源: {r['primary_source']}\n")
                f.write(f"  置信度: {r['confidence']}\n")
                f.write(f"  主要问题: {r['issues'][0] if r['issues'] else '无'}\n")
                f.write(f"{'-' * 60}\n")

        print(f"📊 汇总报告已保存: {summary_path}")


if __name__ == "__main__":
    # 使用示例
    DATA_PATH = "./intel_robotic_welding_dataset/"
    ANOMALY_CSV = "anomalous_files_detailed.csv"

    if not os.path.exists(DATA_PATH):
        print("❌ 数据集路径不存在")
        exit()

    # 创建诊断器
    diagnoser = AnomalyDiagnoser(DATA_PATH)

    # 方式1: 诊断单个文件
    # diagnoser.diagnose_sample("10-04-22-0004-04.avi")

    # 方式2: 批量诊断top 10异常样本
    if os.path.exists(ANOMALY_CSV):
        diagnoser.batch_diagnose(ANOMALY_CSV, top_n=15, save_dir='diagnosis_results')
    else:
        print(f"❌ 找不到异常文件列表: {ANOMALY_CSV}")
        print("请先运行 range.py 生成异常文件列表")