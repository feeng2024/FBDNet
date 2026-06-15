import torch
import numpy as np
import matplotlib.pyplot as plt

# 让 PDF 中字体更稳定，便于投稿系统和编辑器识别
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from sklearn.manifold import TSNE
import seaborn as sns
import os
import torch.nn.functional as F


def plot_dual_branch_tsne(base_dir, prefix, level='d1', sample_size=2000):
    fg_path = os.path.join(base_dir, f"{prefix}_{level}.pt")
    bg_path = os.path.join(base_dir, f"{prefix}_{level}_B.pt")
    gt_path = os.path.join(base_dir, f"{prefix}_gt.pt")

    print(f"[*] Loading data for {prefix} (Level: {level}) from {base_dir}...")

    if not (os.path.exists(fg_path) and os.path.exists(bg_path) and os.path.exists(gt_path)):
        print(f"[!] 报错：找不到文件！请检查路径：\n- {fg_path}\n- {bg_path}\n- {gt_path}")
        return

    # 1. 加载双分支特征和 GT
    fg_feat = torch.load(fg_path)
    bg_feat = torch.load(bg_path)
    gts = torch.load(gt_path)

    if len(gts.shape) == 3:
        gts = gts.unsqueeze(1)  # 确保 GT 是 [N, 1, H, W] 格式

    # 动态匹配空间分辨率
    if gts.shape[2:] != fg_feat.shape[2:]:
        gts = F.interpolate(gts.float(), size=fg_feat.shape[2:], mode='nearest')

    C = fg_feat.shape[1]

    # 2. 展平
    fg_flat = fg_feat.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    bg_flat = bg_feat.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    gts_flat = gts.reshape(-1).numpy()

    # 3. 严格筛选前景/背景像素
    fg_mask = gts_flat > 0.5
    bg_mask = gts_flat <= 0.5

    valid_fg_features = fg_flat[fg_mask]
    valid_bg_features = bg_flat[bg_mask]

    # 4. 随机抽样
    num_fg = len(valid_fg_features)
    num_bg = len(valid_bg_features)

    if num_fg == 0 or num_bg == 0:
        print(f"[!] 跳过 {prefix} ({level})：前景或背景有效像素数量为 0。")
        return

    fg_sample_idx = np.random.choice(num_fg, min(sample_size, num_fg), replace=False)
    bg_sample_idx = np.random.choice(num_bg, min(sample_size, num_bg), replace=False)

    fg_sampled = valid_fg_features[fg_sample_idx]
    bg_sampled = valid_bg_features[bg_sample_idx]

    # 合并
    X = np.vstack((fg_sampled, bg_sampled))
    y = np.hstack((np.ones(len(fg_sampled)), np.zeros(len(bg_sampled))))

    print(f"[*] Running t-SNE on {X.shape[0]} points... (This may take a minute)")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42,
        init='pca'
    )
    X_tsne = tsne.fit_transform(X)

    # 5. 绘图
    plt.figure(figsize=(8, 6))
    palette = {0: '#3498db', 1: '#e74c3c'}

    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y,
        palette=palette,
        s=20,
        alpha=0.7,
        edgecolor=None
    )

    # 去掉 title，其他保持不变
    # plt.title(f"Feature Space: {prefix.upper()} ({level})", fontsize=16, fontweight='bold')

    plt.legend(
        title='Features',
        labels=[f'Background ({level}_B)', f'Foreground ({level})'],
        loc='best'
    )

    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)

    # 6. 保存为 PDF
    out_dir = './Result/wo_tri_loss/TSNE/'
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{prefix}_{level}_tsne.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"[+] Saved successfully to {out_path}\n")


if __name__ == "__main__":

    CUSTOM_FEATURE_DIR = './Checkpoints/Loss_xiaorong/wo tri_loss/CAMO/zhengchang'
    prefixes = ['zhengchang_CAMO']

    levels = ['d1', 'd2', 'd3']

    for p in prefixes:
        for lvl in levels:
            try:
                plot_dual_branch_tsne(
                    base_dir=CUSTOM_FEATURE_DIR,
                    prefix=p,
                    level=lvl
                )
            except Exception as e:
                print(f"[!] 处理 {lvl} 时出错: {e}")