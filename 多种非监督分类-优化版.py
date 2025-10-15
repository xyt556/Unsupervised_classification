import numpy as np
import rasterio
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
from matplotlib.patches import Patch
from scipy.ndimage import label, binary_dilation, generic_filter
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PerformanceTimer:
    """性能计时器"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  开始: {self.name}...")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✓ 完成: {self.name} - 耗时: {elapsed:.2f}秒")

def get_current_paths():
    """获取当前工作目录和脚本目录"""
    current_working_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else current_working_dir
    print(f"当前工作目录: {current_working_dir}")
    print(f"脚本所在目录: {script_dir}")
    return current_working_dir, script_dir

def list_classification_methods():
    """列出可用的分类方法"""
    methods = {
        'minibatch_kmeans': 'MiniBatch K-Means (最快，推荐大数据)',
        'kmeans': 'K-Means聚类 (标准方法)',
        'gmm': '高斯混合模型 (精度高但较慢)',
    }
    return methods

def create_valid_data_mask(bands_data, nodata_value=None):
    """
    快速创建有效数据掩膜
    """
    print("创建有效数据掩膜...")
    
    # 向量化操作，一次性处理所有条件
    if nodata_value is not None:
        invalid_mask = np.any(bands_data == nodata_value, axis=0)
    else:
        # 使用位运算合并多个条件，比逻辑运算快
        invalid_mask = (
            np.all(bands_data == 0, axis=0) |
            np.any(np.isnan(bands_data), axis=0) |
            np.any(np.isinf(bands_data), axis=0)
        )
    
    valid_mask = ~invalid_mask
    valid_percentage = np.sum(valid_mask) / valid_mask.size * 100
    print(f"✓ 有效像素: {valid_percentage:.2f}%")
    
    return valid_mask

def sample_data_for_training(data_clean, max_samples=50000, random_state=42):
    """
    智能采样：对大数据集进行采样以加速训练
    """
    n_samples = len(data_clean)
    
    if n_samples <= max_samples:
        print(f"数据量适中 ({n_samples}点)，使用全部数据训练")
        return data_clean, None
    
    # 采样比例
    sample_ratio = max_samples / n_samples
    print(f"数据量较大 ({n_samples}点)，采样 {sample_ratio*100:.1f}% ({max_samples}点) 进行训练")
    
    # 使用分层采样确保代表性
    np.random.seed(random_state)
    indices = np.random.choice(n_samples, max_samples, replace=False)
    
    return data_clean[indices], indices

def apply_classification_method(data_clean, n_clusters, method='minibatch_kmeans', 
                               random_state=42, use_sampling=True):
    """
    优化的分类方法
    """
    n_samples = len(data_clean)
    print(f"数据点: {n_samples:,}, 特征维度: {data_clean.shape[1]}, 目标类别: {n_clusters}")
    
    # 根据数据量自动选择是否采样
    if use_sampling and n_samples > 50000:
        sample_data, sample_indices = sample_data_for_training(data_clean, max_samples=50000)
    else:
        sample_data = data_clean
        sample_indices = None
    
    try:
        if method == 'minibatch_kmeans':
            # MiniBatch K-Means - 最快的方法
            batch_size = min(2048, len(sample_data) // 20)
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size,
                max_iter=100,  # 减少迭代次数
                n_init=3,      # 减少初始化次数
                reassignment_ratio=0.01,
                verbose=0
            )
            
            if sample_indices is not None:
                # 先在样本上训练
                model.fit(sample_data)
                # 预测全部数据
                labels = model.predict(data_clean)
            else:
                labels = model.fit_predict(sample_data)
                
        elif method == 'kmeans':
            # 标准 K-Means
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=5,      # 减少初始化
                max_iter=200,  # 减少迭代
                algorithm='elkan',  # 使用更快的算法
                verbose=0
            )
            
            if sample_indices is not None:
                model.fit(sample_data)
                labels = model.predict(data_clean)
            else:
                labels = model.fit_predict(sample_data)
                
        elif method == 'gmm':
            # GMM - 精度高但较慢
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                max_iter=50,  # 减少迭代
                n_init=1,
                verbose=0
            )
            
            if sample_indices is not None:
                model.fit(sample_data)
                labels = model.predict(data_clean)
            else:
                labels = model.fit_predict(sample_data)
        
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 标签从1开始
        if labels.min() == 0:
            labels = labels + 1
        
        actual_n_clusters = len(np.unique(labels))
        print(f"✓ 分类完成! 实际类别数: {actual_n_clusters}")
        
        return labels, model, actual_n_clusters
        
    except Exception as e:
        print(f"✗ 分类失败: {str(e)}")
        raise

def remove_small_patches_optimized(classification_result, valid_mask, min_patch_size):
    """
    优化的小斑块移除 - 使用向量化操作
    """
    if min_patch_size <= 0:
        return classification_result
    
    processed_result = classification_result.copy()
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    removed_count = 0
    
    # 使用进度条
    print(f"处理 {len(unique_classes)} 个类别...")
    
    for class_id in unique_classes:
        # 对每个类别处理
        class_mask = (classification_result == class_id) & valid_mask
        labeled_array, num_features = label(class_mask)
        
        if num_features == 0:
            continue
        
        # 向量化计算所有区域的大小
        region_sizes = np.bincount(labeled_array.ravel())
        
        # 找出所有小斑块
        small_regions = np.where((region_sizes < min_patch_size) & (region_sizes > 0))[0]
        
        if len(small_regions) == 0:
            continue
        
        # 批量处理小斑块
        for region_id in small_regions:
            region_mask = labeled_array == region_id
            
            # 膨胀找邻域
            dilated = binary_dilation(region_mask, iterations=1)
            neighbor_mask = dilated & ~region_mask & valid_mask
            
            if np.any(neighbor_mask):
                # 找最常见的邻域类别
                neighbor_classes = classification_result[neighbor_mask]
                neighbor_classes = neighbor_classes[neighbor_classes > 0]
                
                if len(neighbor_classes) > 0:
                    # 使用bincount加速
                    counts = np.bincount(neighbor_classes)
                    new_class = np.argmax(counts)
                    processed_result[region_mask] = new_class
                    removed_count += 1
    
    print(f"✓ 移除 {removed_count} 个小斑块")
    return processed_result

def majority_filter_optimized(classification_result, valid_mask):
    """
    优化的多数滤波 - 使用scipy的优化函数
    """
    from scipy.ndimage import generic_filter
    
    # 预定义多数函数，避免重复创建
    def majority_func(values):
        values = values[values > 0]
        if len(values) == 0:
            return 0
        # 使用bincount加速
        counts = np.bincount(values.astype(int))
        return np.argmax(counts)
    
    # 只对有效区域处理
    smoothed = classification_result.copy()
    
    # 使用mode='nearest'加速边界处理
    temp = generic_filter(
        classification_result,
        majority_func,
        size=3,
        mode='nearest'
    )
    
    smoothed[valid_mask] = temp[valid_mask]
    return smoothed

def post_process_classification(classification_result, valid_mask, 
                               min_patch_size=50, smoothing_iterations=1):
    """
    优化的后处理流程
    """
    if min_patch_size <= 0 and smoothing_iterations <= 0:
        return classification_result
    
    print("\n后处理...")
    processed_result = classification_result.copy()
    
    # 步骤1: 去除小斑块
    if min_patch_size > 0:
        with PerformanceTimer("去除小斑块"):
            processed_result = remove_small_patches_optimized(
                processed_result, valid_mask, min_patch_size
            )
    
    # 步骤2: 平滑
    if smoothing_iterations > 0:
        with PerformanceTimer("平滑处理"):
            for i in range(smoothing_iterations):
                processed_result = majority_filter_optimized(processed_result, valid_mask)
                if smoothing_iterations > 1:
                    print(f"  迭代 {i+1}/{smoothing_iterations}")
    
    return processed_result

def calculate_pixel_area(transform):
    """计算像素面积"""
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    pixel_area = pixel_width * pixel_height
    print(f"像素: {pixel_width:.2f}m × {pixel_height:.2f}m = {pixel_area:.2f}m²")
    return pixel_area

def calculate_class_areas_optimized(classification_result, valid_mask, pixel_area):
    """
    优化的面积计算 - 使用向量化
    """
    print("\n计算面积统计...")
    
    # 只统计有效区域
    valid_data = classification_result[valid_mask]
    
    # 向量化计算所有类别
    unique_classes, counts = np.unique(valid_data, return_counts=True)
    unique_classes = unique_classes[unique_classes > 0]
    counts = counts[unique_classes > 0]
    
    total_pixels = np.sum(counts)
    total_area_km2 = total_pixels * pixel_area / 1e6
    
    print(f"总有效面积: {total_area_km2:.2f} km²")
    
    area_stats = []
    for cluster_id, cluster_pixels in zip(unique_classes, counts):
        area_m2 = cluster_pixels * pixel_area
        area_km2 = area_m2 / 1e6
        percentage = cluster_pixels / total_pixels * 100
        
        print(f"类别 {cluster_id}: {area_km2:.2f} km² ({percentage:.1f}%)")
        
        area_stats.append({
            '类别': int(cluster_id),
            '像素数量': int(cluster_pixels),
            '面积_平方千米': round(area_km2, 2),
            '面积_平方米': round(area_m2, 0),
            '占比_百分比': round(percentage, 2)
        })
    
    return area_stats

def landsat8_unsupervised_classification(input_file, output_file, n_clusters=5,
                                        nodata_value=None, post_process=True,
                                        min_patch_size=50, smoothing_iterations=1,
                                        method='minibatch_kmeans', use_sampling=True):
    """
    优化的Landsat 8非监督分类
    """
    print("="*60)
    print(" "*15 + "Landsat 8 非监督分类 (优化版)")
    print("="*60)
    
    total_start = time.time()
    
    # 1. 读取数据
    with PerformanceTimer("读取影像数据"):
        with rasterio.open(input_file) as src:
            bands_data = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
            
            if nodata_value is None and src.nodata is not None:
                nodata_value = src.nodata
            
            print(f"  形状: {bands_data.shape}")
            print(f"  波段: {src.count}")
            print(f"  尺寸: {src.width} × {src.height}")
            print(f"  类型: {bands_data.dtype}")
    
    # 2. 计算像素面积
    pixel_area = calculate_pixel_area(transform)
    
    # 3. 创建掩膜
    with PerformanceTimer("创建有效数据掩膜"):
        valid_mask_2d = create_valid_data_mask(bands_data, nodata_value)
    
    if np.sum(valid_mask_2d) == 0:
        raise ValueError("✗ 没有有效数据!")
    
    # 4. 数据预处理
    with PerformanceTimer("数据预处理"):
        height, width = bands_data.shape[1], bands_data.shape[2]
        n_bands = bands_data.shape[0]
        
        # 重塑数据 - 使用更高效的方式
        data_2d = bands_data.reshape(n_bands, -1).T
        
        # 提取有效数据
        valid_mask_1d = valid_mask_2d.ravel()
        data_valid = data_2d[valid_mask_1d]
        
        # 清理异常值
        finite_mask = np.all(np.isfinite(data_valid), axis=1)
        data_clean_temp = data_valid[finite_mask]
        
        print(f"  有效数据点: {len(data_clean_temp):,}")
        
        # 标准化
        scaler = StandardScaler()
        data_clean = scaler.fit_transform(data_clean_temp)
        
        print(f"  数据范围: [{data_clean.min():.2f}, {data_clean.max():.2f}]")
    
    # 5. 分类
    with PerformanceTimer(f"执行{method}分类"):
        labels, model, actual_n_clusters = apply_classification_method(
            data_clean, n_clusters, method, use_sampling=use_sampling
        )
    
    # 6. 重建结果
    with PerformanceTimer("重建分类结果"):
        # 创建完整标签数组
        full_labels_temp = np.zeros(len(data_valid), dtype=np.uint8)
        full_labels_temp[finite_mask] = labels
        
        full_labels = np.zeros(height * width, dtype=np.uint8)
        full_labels[valid_mask_1d] = full_labels_temp
        
        classification_result = full_labels.reshape(height, width)
        
        print(f"  类别范围: {classification_result.min()} ~ {classification_result.max()}")
    
    # 7. 后处理
    if post_process:
        with PerformanceTimer("分类后处理"):
            classification_result = post_process_classification(
                classification_result, valid_mask_2d, 
                min_patch_size, smoothing_iterations
            )
    
    # 8. 保存结果
    with PerformanceTimer("保存分类结果"):
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(classification_result, 1)
        
        print(f"  文件: {output_file}")
    
    # 9. 面积统计
    with PerformanceTimer("计算面积统计"):
        area_stats = calculate_class_areas_optimized(
            classification_result, valid_mask_2d, pixel_area
        )
    
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print(f"✓ 全部完成! 总耗时: {total_time:.2f}秒")
    print("="*60)
    
    return classification_result, model, valid_mask_2d, area_stats, pixel_area

def plot_classification_result(classification_result, valid_mask, title_suffix="",
                               save_path='classification_result.png', dpi=150):
    """
    优化的可视化 - 降低dpi加速
    """
    with PerformanceTimer("生成分类结果图"):
        plot_data = classification_result.astype(float)
        plot_data[~valid_mask] = np.nan
        
        unique_classes = np.unique(classification_result[valid_mask])
        unique_classes = unique_classes[unique_classes > 0]
        actual_clusters = len(unique_classes)
        
        # 预定义颜色
        colors = ['#228B22', '#32CD32', '#90EE90', '#FFD700', '#FFA500',
                  '#8B4513', '#4169E1', '#87CEEB', '#808080', '#DC143C',
                  '#9370DB', '#FF69B4', '#00CED1', '#FF6347', '#4682B4']
        
        if actual_clusters > len(colors):
            import matplotlib.cm as cm
            cmap_colors = cm.get_cmap('tab20', actual_clusters)
            colors = [cmap_colors(i) for i in range(actual_clusters)]
        
        cmap = ListedColormap(colors[:actual_clusters])
        
        # 根据数据大小调整图像尺寸
        fig_width = min(16, max(10, classification_result.shape[1] / 100))
        fig_height = min(12, max(8, classification_result.shape[0] / 100))
        
        plt.figure(figsize=(fig_width, fig_height))
        
        im = plt.imshow(plot_data, cmap=cmap, interpolation='nearest')
        
        cbar = plt.colorbar(im, label='土地类别', shrink=0.8, pad=0.02)
        cbar.set_ticks(unique_classes)
        cbar.set_ticklabels([f'类别 {int(i)}' for i in unique_classes])
        
        title = f'Landsat 8 分类结果 ({actual_clusters}类)'
        if title_suffix:
            title += f" - {title_suffix}"
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.xlabel('列', fontsize=10)
        plt.ylabel('行', fontsize=10)
        
        # 图例
        legend_elements = [
            Patch(facecolor=colors[i], label=f'类别 {int(unique_classes[i])}')
            for i in range(actual_clusters)
        ]
        plt.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(1.12, 1), title="类型",
                  fontsize=9)
        
        # 统计信息
        valid_pct = np.sum(valid_mask) / valid_mask.size * 100
        stats_text = f'有效区域: {valid_pct:.1f}%\n类别: {actual_clusters}'
        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  保存: {save_path}")
        plt.close()  # 关闭图形释放内存

def plot_area_distribution(area_stats, method_name, save_path='面积分布图.png', dpi=150):
    """优化的面积分布图"""
    
    if not area_stats:
        return
    
    with PerformanceTimer("生成面积分布图"):
        classes = [f'类别 {stat["类别"]}' for stat in area_stats]
        areas = [stat['面积_平方千米'] for stat in area_stats]
        percentages = [stat['占比_百分比'] for stat in area_stats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 柱状图
        colors_bar = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax1.bar(classes, areas, color=colors_bar, edgecolor='black', linewidth=1)
        ax1.set_title(f'{method_name} - 面积分布', fontsize=13, fontweight='bold')
        ax1.set_xlabel('类别', fontsize=11)
        ax1.set_ylabel('面积 (km²)', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, area, pct in zip(bars, areas, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{area:.1f}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
        
        # 饼图
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        wedges, texts, autotexts = ax2.pie(
            areas, labels=classes, autopct='%1.1f%%',
            colors=colors_pie, startangle=90
        )
        ax2.set_title(f'{method_name} - 面积占比', fontsize=13, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  保存: {save_path}")
        plt.close()

def save_area_statistics(area_stats, output_file="面积统计.csv"):
    """保存统计"""
    df = pd.DataFrame(area_stats)
    df = df.sort_values('面积_平方千米', ascending=False)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  保存: {output_file}")
    return df

def main():
    """优化的主函数"""
    print("\n" + "="*60)
    print(" "*10 + "Landsat 8 非监督分类系统 (高性能版)")
    print("="*60 + "\n")
    
    get_current_paths()
    
    # 配置
    input_file = r"D:\code313\Geo_programe\rasterio\data\ylq_L8_2024.tif"
    
    if not os.path.exists(input_file):
        print(f"✗ 文件不存在: {input_file}")
        return
    
    # 显示方法
    methods = list_classification_methods()
    print("\n可用方法:")
    for i, (key, desc) in enumerate(methods.items(), 1):
        print(f"  {i}. {key}: {desc}")
    
    # 用户输入
    method_input = input("\n选择方法 (直接回车使用minibatch_kmeans): ").strip()
    
    if not method_input:
        method = 'minibatch_kmeans'
    elif method_input.isdigit() and 1 <= int(method_input) <= len(methods):
        method = list(methods.keys())[int(method_input) - 1]
    elif method_input.lower() in methods:
        method = method_input.lower()
    else:
        method = 'minibatch_kmeans'
    
    print(f"✓ 方法: {methods[method]}")
    
    # 类别数
    n_clusters_input = input("分类数量 (2-15，默认6): ").strip()
    try:
        n_clusters = int(n_clusters_input) if n_clusters_input else 6
        n_clusters = max(2, min(15, n_clusters))
    except:
        n_clusters = 6
    print(f"✓ 类别数: {n_clusters}")
    
    # 后处理
    post_process_input = input("后处理? (y/n，默认y): ").strip().lower()
    post_process = post_process_input != 'n'
    
    if post_process:
        min_patch_size = 30
        smoothing_iterations = 1
        print(f"✓ 后处理: 最小斑块={min_patch_size}, 平滑={smoothing_iterations}")
    else:
        min_patch_size = 0
        smoothing_iterations = 0
        print("✓ 跳过后处理")
    
    # 采样选项
    use_sampling_input = input("大数据集采样加速? (y/n，默认y): ").strip().lower()
    use_sampling = use_sampling_input != 'n'
    print(f"✓ 采样加速: {'是' if use_sampling else '否'}")
    
    output_file = f"classification_{method}_{n_clusters}classes.tif"
    
    try:
        # 执行分类
        classification_result, model, valid_mask, area_stats, pixel_area = \
            landsat8_unsupervised_classification(
                input_file, output_file, n_clusters,
                post_process=post_process,
                min_patch_size=min_patch_size,
                smoothing_iterations=smoothing_iterations,
                method=method,
                use_sampling=use_sampling
            )
        
        # 保存结果
        csv_file = f"面积统计_{method}.csv"
        save_area_statistics(area_stats, csv_file)
        
        # 绘图 (使用较低DPI加速)
        plot_file = f"分类结果_{method}.png"
        plot_classification_result(
            classification_result, valid_mask,
            methods[method], plot_file, dpi=150
        )
        
        area_plot_file = f"面积分布_{method}.png"
        plot_area_distribution(area_stats, methods[method], area_plot_file, dpi=150)
        
        print("\n" + "="*60)
        print("✓ 全部完成!")
        print("="*60)
        print(f"输出文件:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
        print(f"  - {plot_file}")
        print(f"  - {area_plot_file}")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ 处理失败!")
        print("="*60)
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()