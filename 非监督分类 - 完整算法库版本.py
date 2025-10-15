import numpy as np
import rasterio
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, OPTICS, 
                             MeanShift, Birch, SpectralClustering, 
                             AgglomerativeClustering)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
from matplotlib.patches import Patch
from scipy.ndimage import label, binary_dilation, generic_filter
from scipy.ndimage import maximum_filter, minimum_filter
from scipy import ndimage
import pandas as pd
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

# 尝试导入numba加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  numba未安装，部分功能将较慢。安装: pip install numba")

# 尝试导入scikit-fuzzy
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
    print("✓ scikit-fuzzy 已安装")
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️  scikit-fuzzy未安装，将使用简化版模糊C均值")

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
        print(f"\n⏱️  {self.name}...")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✓ 完成 - 耗时: {elapsed:.2f}秒")

# ==================== 优化的后处理函数 ====================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _majority_filter_numba(data, output, valid_mask, window_size=3):
        """
        Numba加速的多数滤波
        """
        height, width = data.shape
        offset = window_size // 2
        
        for i in prange(offset, height - offset):
            for j in range(offset, width - offset):
                if not valid_mask[i, j]:
                    continue
                
                # 提取窗口
                window = data[i-offset:i+offset+1, j-offset:j+offset+1]
                valid_window = valid_mask[i-offset:i+offset+1, j-offset:j+offset+1]
                
                # 只考虑有效值
                valid_values = window[valid_window]
                valid_values = valid_values[valid_values > 0]
                
                if len(valid_values) == 0:
                    continue
                
                # 找出现次数最多的值
                max_count = 0
                majority_value = data[i, j]
                
                for val in valid_values:
                    count = np.sum(valid_values == val)
                    if count > max_count:
                        max_count = count
                        majority_value = val
                
                output[i, j] = majority_value
    
    @jit(nopython=True, cache=True)
    def _get_neighbor_mode(classification, i, j, valid_mask):
        """获取邻域的众数（最常见值）"""
        height, width = classification.shape
        values = []
        
        # 8邻域
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    if valid_mask[ni, nj] and classification[ni, nj] > 0:
                        values.append(classification[ni, nj])
        
        if len(values) == 0:
            return 0
        
        # 找众数
        max_count = 0
        mode_value = values[0]
        
        for val in values:
            count = sum(1 for v in values if v == val)
            if count > max_count:
                max_count = count
                mode_value = val
        
        return mode_value
else:
    _majority_filter_numba = None
    _get_neighbor_mode = None

def majority_filter_fast(classification_result, valid_mask, window_size=3):
    """
    快速多数滤波 - 使用scipy优化
    """
    if NUMBA_AVAILABLE:
        # 使用numba加速版本
        output = classification_result.copy()
        _majority_filter_numba(classification_result, output, valid_mask, window_size)
        return output
    else:
        # 使用scipy的mode滤波
        from scipy.ndimage import generic_filter
        
        def mode_func(values):
            values = values[values > 0]
            if len(values) == 0:
                return 0
            # 快速众数计算
            unique, counts = np.unique(values, return_counts=True)
            return unique[np.argmax(counts)]
        
        # 只对有效区域滤波
        smoothed = classification_result.copy()
        temp = generic_filter(
            classification_result,
            mode_func,
            size=window_size,
            mode='constant',
            cval=0
        )
        smoothed[valid_mask] = temp[valid_mask]
        
        return smoothed

def remove_small_patches_fast(classification_result, valid_mask, min_patch_size):
    """
    快速移除小斑块 - 优化版
    
    优化策略:
    1. 一次性标记所有连通区域
    2. 批量处理小斑块
    3. 向量化邻域查找
    """
    if min_patch_size <= 0:
        return classification_result
    
    print(f"  快速移除小斑块 (最小尺寸: {min_patch_size})...")
    
    processed_result = classification_result.copy()
    
    # 一次性对所有类别进行连通性分析
    # 使用结构化标签，同时考虑类别值
    labeled_array, num_features = label(valid_mask)
    
    # 为每个有效区域找到其类别
    region_to_class = {}
    region_sizes = {}
    
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        region_classes = classification_result[region_mask]
        
        if len(region_classes) > 0:
            # 找出该区域的主要类别
            unique, counts = np.unique(region_classes, return_counts=True)
            if len(unique) > 0:
                main_class = unique[np.argmax(counts)]
                region_to_class[region_id] = main_class
                region_sizes[region_id] = np.sum(region_mask)
    
    # 对每个类别分别进行连通性分析
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    removed_count = 0
    
    for class_id in unique_classes:
        class_mask = (classification_result == class_id) & valid_mask
        
        if not np.any(class_mask):
            continue
        
        # 对当前类别进行连通性分析
        class_labeled, class_num_features = label(class_mask)
        
        if class_num_features == 0:
            continue
        
        # 向量化计算所有区域的大小
        region_sizes_class = np.bincount(class_labeled.ravel())
        
        # 找出所有小区域
        small_regions = np.where(
            (region_sizes_class < min_patch_size) & 
            (region_sizes_class > 0)
        )[0]
        
        if len(small_regions) == 0:
            continue
        
        # 批量处理小区域
        for region_id in small_regions:
            region_mask = class_labeled == region_id
            
            # 使用形态学膨胀找邻域（更快）
            dilated = binary_dilation(region_mask, iterations=1)
            neighbor_mask = dilated & ~region_mask & valid_mask
            
            if np.any(neighbor_mask):
                # 找邻域最常见的类别
                neighbor_classes = classification_result[neighbor_mask]
                neighbor_classes = neighbor_classes[neighbor_classes > 0]
                
                if len(neighbor_classes) > 0:
                    # 使用bincount加速
                    unique_neighbors, counts = np.unique(neighbor_classes, return_counts=True)
                    new_class = unique_neighbors[np.argmax(counts)]
                    processed_result[region_mask] = new_class
                    removed_count += 1
    
    print(f"  ✓ 移除了 {removed_count} 个小斑块")
    return processed_result

def remove_small_patches_ultra_fast(classification_result, valid_mask, min_patch_size):
    """
    超快速移除小斑块 - 使用分块并行处理
    
    适合大图像
    """
    if min_patch_size <= 0:
        return classification_result
    
    height, width = classification_result.shape
    
    # 如果图像不大，使用标准方法
    if height * width < 10000000:  # 小于1000万像素
        return remove_small_patches_fast(classification_result, valid_mask, min_patch_size)
    
    print(f"  超快速移除小斑块 (分块并行处理)...")
    
    # 分块处理
    block_size = 2048
    processed_result = classification_result.copy()
    
    # 计算需要多少块
    n_blocks_h = (height + block_size - 1) // block_size
    n_blocks_w = (width + block_size - 1) // block_size
    
    total_removed = 0
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            # 计算块的边界（带重叠）
            overlap = 50
            h_start = max(0, i * block_size - overlap)
            h_end = min(height, (i + 1) * block_size + overlap)
            w_start = max(0, j * block_size - overlap)
            w_end = min(width, (j + 1) * block_size + overlap)
            
            # 提取块
            block_class = classification_result[h_start:h_end, w_start:w_end].copy()
            block_valid = valid_mask[h_start:h_end, w_start:w_end]
            
            # 处理块
            block_processed = remove_small_patches_fast(block_class, block_valid, min_patch_size)
            
            # 写回中心区域（不包括重叠部分）
            actual_h_start = i * block_size
            actual_h_end = min(height, (i + 1) * block_size)
            actual_w_start = j * block_size
            actual_w_end = min(width, (j + 1) * block_size)
            
            offset_h = actual_h_start - h_start
            offset_w = actual_w_start - w_start
            
            processed_result[actual_h_start:actual_h_end, actual_w_start:actual_w_end] = \
                block_processed[
                    offset_h:offset_h + (actual_h_end - actual_h_start),
                    offset_w:offset_w + (actual_w_end - actual_w_start)
                ]
    
    print(f"  ✓ 分块处理完成")
    return processed_result

def post_process_classification_fast(classification_result, valid_mask, 
                                     min_patch_size=50, smoothing_iterations=1):
    """
    优化的后处理流程
    
    速度提升策略:
    1. 使用numba加速
    2. 向量化操作
    3. 分块并行处理
    4. 减少内存复制
    """
    if min_patch_size <= 0 and smoothing_iterations <= 0:
        return classification_result
    
    print("\n快速后处理...")
    processed_result = classification_result.copy()
    
    # 步骤1: 去除小斑块
    if min_patch_size > 0:
        with PerformanceTimer("去除小斑块"):
            # 根据数据规模选择方法
            total_pixels = classification_result.size
            if total_pixels > 10_000_000:
                processed_result = remove_small_patches_ultra_fast(
                    processed_result, valid_mask, min_patch_size
                )
            else:
                processed_result = remove_small_patches_fast(
                    processed_result, valid_mask, min_patch_size
                )
    
    # 步骤2: 平滑处理
    if smoothing_iterations > 0:
        with PerformanceTimer("平滑处理"):
            for i in range(smoothing_iterations):
                if smoothing_iterations > 1:
                    print(f"  迭代 {i+1}/{smoothing_iterations}...")
                processed_result = majority_filter_fast(processed_result, valid_mask)
    
    return processed_result

# ==================== 其他函数保持不变 ====================

def list_classification_methods():
    """列出所有可用的分类方法"""
    methods = {
        'minibatch_kmeans': {
            'name': 'MiniBatch K-Means',
            'desc': '最快速的方法，适合大数据集',
            'speed': '⚡⚡⚡⚡⚡',
            'accuracy': '★★★☆☆',
            'need_n_clusters': True,
            'best_for': '大数据集快速分类',
            'available': True
        },
        'kmeans': {
            'name': 'K-Means聚类',
            'desc': '经典聚类算法，速度和精度平衡',
            'speed': '⚡⚡⚡⚡☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': True,
            'best_for': '标准遥感分类',
            'available': True
        },
        'birch': {
            'name': 'Birch聚类',
            'desc': '层次聚类，内存效率高',
            'speed': '⚡⚡⚡⚡☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': True,
            'best_for': '大数据集层次分类',
            'available': True
        },
        'gmm': {
            'name': '高斯混合模型',
            'desc': '概率模型，精度高',
            'speed': '⚡⚡⚡☆☆',
            'accuracy': '★★★★★',
            'need_n_clusters': True,
            'best_for': '精细分类',
            'available': True
        },
        'fuzzy_cmeans': {
            'name': '模糊C均值',
            'desc': '遥感常用，软分类',
            'speed': '⚡⚡⚡☆☆',
            'accuracy': '★★★★★',
            'need_n_clusters': True,
            'best_for': '边界模糊的地物分类',
            'available': FUZZY_AVAILABLE
        },
        'spectral': {
            'name': '谱聚类',
            'desc': '基于图论，处理复杂结构',
            'speed': '⚡⚡☆☆☆',
            'accuracy': '★★★★★',
            'need_n_clusters': True,
            'best_for': '复杂空间结构',
            'available': True
        },
        'dbscan': {
            'name': 'DBSCAN',
            'desc': '基于密度，自动确定类别数',
            'speed': '⚡⚡⚡☆☆',
            'accuracy': '★★★☆☆',
            'need_n_clusters': False,
            'best_for': '噪声数据处理',
            'available': True
        },
        'optics': {
            'name': 'OPTICS',
            'desc': 'DBSCAN改进版，处理变密度',
            'speed': '⚡⚡☆☆☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': False,
            'best_for': '变密度数据',
            'available': True
        },
        'meanshift': {
            'name': 'Mean Shift',
            'desc': '均值漂移，自动确定类别',
            'speed': '⚡⚡☆☆☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': False,
            'best_for': '自然聚类发现',
            'available': True
        },
        'isodata': {
            'name': 'ISODATA',
            'desc': '遥感经典算法，可变类别数',
            'speed': '⚡⚡⚡☆☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': True,
            'best_for': '遥感影像标准分类',
            'available': True
        },
        'hierarchical': {
            'name': '层次聚类',
            'desc': '凝聚层次聚类',
            'speed': '⚡⚡☆☆☆',
            'accuracy': '★★★★☆',
            'need_n_clusters': True,
            'best_for': '中小数据集',
            'available': True
        },
    }
    return methods

def print_methods_table():
    """打印方法对比表"""
    methods = list_classification_methods()
    
    print("\n" + "="*105)
    print(f"{'序号':<4} {'方法名':<22} {'速度':<15} {'精度':<15} {'适用场景':<30} {'状态':<8}")
    print("="*105)
    
    idx = 1
    for key, info in methods.items():
        if info['available']:
            status = "✓可用"
            print(f"{idx:<4} {info['name']:<22} {info['speed']:<15} {info['accuracy']:<15} {info['best_for']:<30} {status:<8}")
            idx += 1
        else:
            status = "✗不可用"
            print(f"{'--':<4} {info['name']:<22} {info['speed']:<15} {info['accuracy']:<15} {info['best_for']:<30} {status:<8}")
    
    print("="*105)
    print("说明: ⚡=速度 ★=精度 | 推荐: 大数据用MiniBatch, 高精度用GMM, 遥感用ISODATA")
    if NUMBA_AVAILABLE:
        print("✓ Numba加速已启用 - 后处理将显著加快")
    else:
        print("💡 提示: 安装numba可大幅加速后处理 (pip install numba)")
    print("="*105)

def create_valid_data_mask(bands_data, nodata_value=None):
    """创建有效数据掩膜"""
    print("创建有效数据掩膜...")
    
    if nodata_value is not None:
        invalid_mask = np.any(bands_data == nodata_value, axis=0)
    else:
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
    """智能采样"""
    n_samples = len(data_clean)
    
    if n_samples <= max_samples:
        print(f"使用全部数据 ({n_samples:,}点)")
        return data_clean, None
    
    print(f"采样 {max_samples:,}/{n_samples:,} 点训练")
    np.random.seed(random_state)
    indices = np.random.choice(n_samples, max_samples, replace=False)
    
    return data_clean[indices], indices

def calculate_pixel_area(transform, crs=None):
    """计算单个像素的面积（平方米）"""
    print("\n" + "="*60)
    print("像素面积计算")
    print("="*60)
    
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    
    print(f"像素宽度: {pixel_width}")
    print(f"像素高度: {pixel_height}")
    
    if crs is not None:
        print(f"坐标系统: {crs}")
        
        if crs.is_geographic:
            print("⚠️  检测到地理坐标系（度），需要转换为米")
            pixel_width_m = pixel_width * 111320
            pixel_height_m = pixel_height * 110540
            pixel_area = pixel_width_m * pixel_height_m
        else:
            print("✓ 检测到投影坐标系（米）")
            pixel_area = pixel_width * pixel_height
    else:
        print("⚠️  未提供坐标系信息，假设单位为米")
        pixel_area = pixel_width * pixel_height
    
    print(f"\n单个像素面积: {pixel_area:.2f} 平方米")
    print(f"           = {pixel_area/10000:.6f} 公顷")
    print(f"           = {pixel_area/1000000:.8f} 平方千米")
    print("="*60)
    
    return pixel_area

def calculate_class_areas_enhanced(classification_result, valid_mask, pixel_area, crs=None):
    """增强的面积计算函数"""
    print("\n" + "="*80)
    print("面积统计 (多单位)")
    print("="*80)
    
    if pixel_area <= 0 or not np.isfinite(pixel_area):
        print(f"⚠️  错误: 像素面积无效 (pixel_area = {pixel_area})")
        pixel_area = 900.0
    
    print(f"使用的像素面积: {pixel_area:.2f} 平方米")
    
    valid_data = classification_result[valid_mask]
    
    if len(valid_data) == 0:
        print("⚠️  警告: 没有有效数据用于统计")
        return []
    
    unique_classes, counts = np.unique(valid_data, return_counts=True)
    valid_indices = (unique_classes > 0) & np.isfinite(unique_classes) & (counts > 0)
    unique_classes = unique_classes[valid_indices]
    counts = counts[valid_indices]
    
    if len(unique_classes) == 0:
        print("⚠️  警告: 没有有效的分类类别")
        return []
    
    total_pixels = np.sum(counts)
    total_area_m2 = float(total_pixels) * pixel_area
    total_area_km2 = total_area_m2 / 1_000_000
    total_area_ha = total_area_m2 / 10_000
    
    print(f"\n总有效区域:")
    print(f"  像素数量: {total_pixels:,}")
    print(f"  面积: {total_area_m2:,.2f} 平方米")
    print(f"      = {total_area_km2:,.4f} 平方千米")
    print(f"      = {total_area_ha:,.2f} 公顷")
    
    print("\n" + "-"*80)
    print(f"{'类别':<8} {'像素数':<12} {'平方米':<15} {'平方千米':<12} {'公顷':<12} {'占比%':<10}")
    print("-"*80)
    
    area_stats = []
    
    for cluster_id, cluster_pixels in zip(unique_classes, counts):
        if not np.isfinite(cluster_pixels) or cluster_pixels <= 0:
            continue
        
        area_m2 = float(cluster_pixels) * pixel_area
        area_km2 = area_m2 / 1_000_000
        area_ha = area_m2 / 10_000
        percentage = (float(cluster_pixels) / total_pixels) * 100
        
        if not all(np.isfinite([area_m2, area_km2, area_ha, percentage])):
            continue
        
        print(f"{int(cluster_id):<8} {int(cluster_pixels):<12,} {area_m2:<15,.2f} {area_km2:<12,.4f} {area_ha:<12,.2f} {percentage:<10,.2f}")
        
        area_stats.append({
            '类别': int(cluster_id),
            '像素数量': int(cluster_pixels),
            '面积_平方米': round(float(area_m2), 2),
            '面积_平方千米': round(float(area_km2), 6),
            '面积_公顷': round(float(area_ha), 2),
            '占比_百分比': round(float(percentage), 2)
        })
    
    print("="*80)
    
    return area_stats

# [分类算法函数保持不变 - 这里省略以节省空间]

def apply_kmeans(data, n_clusters, use_sampling):
    """K-Means聚类"""
    sample_data, sample_indices = sample_data_for_training(data) if use_sampling else (data, None)
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=5,
        max_iter=200,
        algorithm='elkan'
    )
    
    if sample_indices is not None:
        model.fit(sample_data)
        labels = model.predict(data)
    else:
        labels = model.fit_predict(data)
    
    return labels, model

def apply_minibatch_kmeans(data, n_clusters, use_sampling):
    """MiniBatch K-Means"""
    batch_size = min(2048, len(data) // 20)
    
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=batch_size,
        max_iter=100,
        n_init=3
    )
    
    labels = model.fit_predict(data)
    return labels, model

def apply_gmm(data, n_clusters, use_sampling):
    """高斯混合模型"""
    sample_data, sample_indices = sample_data_for_training(data, 30000) if use_sampling else (data, None)
    
    model = GaussianMixture(
        n_components=n_clusters,
        random_state=42,
        max_iter=50,
        covariance_type='diag'
    )
    
    if sample_indices is not None:
        model.fit(sample_data)
        labels = model.predict(data)
    else:
        labels = model.fit_predict(data)
    
    return labels, model

def apply_fuzzy_cmeans(data, n_clusters, use_sampling):
    """模糊C均值聚类"""
    print("  使用模糊C均值算法 (遥感经典方法)")
    
    sample_data, sample_indices = sample_data_for_training(data, 20000) if use_sampling else (data, None)
    
    if FUZZY_AVAILABLE:
        data_T = sample_data.T
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_T,
            c=n_clusters,
            m=2,
            error=0.005,
            maxiter=100,
            init=None
        )
        
        if sample_indices is not None:
            u_full, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                data.T, cntr, 2, error=0.005, maxiter=100
            )
            labels = np.argmax(u_full, axis=0)
        else:
            labels = np.argmax(u, axis=0)
        
        print(f"  模糊分割系数 (FPC): {fpc:.3f}")
        
        class FuzzyCMeansModel:
            def __init__(self, centers):
                self.cluster_centers_ = centers
        
        return labels, FuzzyCMeansModel(cntr)
    else:
        print("  警告: scikit-fuzzy未安装，使用KMeans替代")
        return apply_kmeans(data, n_clusters, use_sampling)

def apply_spectral(data, n_clusters, use_sampling):
    """谱聚类"""
    print("  警告: 谱聚类计算量大，强制采样")
    
    max_samples = min(10000, len(data))
    sample_data, sample_indices = sample_data_for_training(data, max_samples)
    
    model = SpectralClustering(
        n_clusters=n_clusters,
        random_state=42,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans'
    )
    
    sample_labels = model.fit_predict(sample_data)
    
    print("  使用KMeans扩展分类结果...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    
    centers = []
    for i in range(n_clusters):
        mask = sample_labels == i
        if np.sum(mask) > 0:
            centers.append(np.mean(sample_data[mask], axis=0))
        else:
            centers.append(sample_data[np.random.randint(len(sample_data))])
    
    kmeans.cluster_centers_ = np.array(centers)
    labels = kmeans.predict(data)
    
    return labels, kmeans

def apply_dbscan(data, use_sampling):
    """DBSCAN"""
    print("  DBSCAN自动确定类别数")
    
    sample_data, sample_indices = sample_data_for_training(data, 10000) if use_sampling else (data, None)
    
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(sample_data if sample_indices is not None else data)
    distances, indices = neighbors.kneighbors(sample_data if sample_indices is not None else data)
    distances = np.sort(distances[:, -1])
    eps = np.percentile(distances, 90) * 0.5
    
    print(f"  自动估计 eps={eps:.3f}")
    
    model = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
    
    if sample_indices is not None:
        sample_labels = model.fit_predict(sample_data)
        
        from sklearn.neighbors import KNeighborsClassifier
        valid_mask = sample_labels >= 0
        if np.sum(valid_mask) > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(sample_data[valid_mask], sample_labels[valid_mask])
            labels = knn.predict(data)
        else:
            labels = np.zeros(len(data), dtype=int)
    else:
        labels = model.fit_predict(data)
    
    labels[labels == -1] = labels.max() + 1
    n_clusters = len(np.unique(labels))
    print(f"  检测到 {n_clusters} 个类别")
    
    return labels, model

def apply_optics(data, use_sampling):
    """OPTICS"""
    print("  OPTICS自动确定类别数")
    
    sample_data, sample_indices = sample_data_for_training(data, 8000) if use_sampling else (data, None)
    
    model = OPTICS(min_samples=10, max_eps=2.0, n_jobs=-1)
    
    if sample_indices is not None:
        sample_labels = model.fit_predict(sample_data)
        
        from sklearn.neighbors import KNeighborsClassifier
        valid_mask = sample_labels >= 0
        if np.sum(valid_mask) > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(sample_data[valid_mask], sample_labels[valid_mask])
            labels = knn.predict(data)
        else:
            labels = np.zeros(len(data), dtype=int)
    else:
        labels = model.fit_predict(data)
    
    labels[labels == -1] = labels.max() + 1
    n_clusters = len(np.unique(labels))
    print(f"  检测到 {n_clusters} 个类别")
    
    return labels, model

def apply_meanshift(data, use_sampling):
    """Mean Shift"""
    print("  Mean Shift自动确定类别数")
    print("  警告: 计算量大，强制采样")
    
    sample_data, _ = sample_data_for_training(data, 5000)
    
    from sklearn.cluster import estimate_bandwidth
    bandwidth = estimate_bandwidth(sample_data, quantile=0.2, n_samples=1000)
    
    print(f"  自动估计 bandwidth={bandwidth:.3f}")
    
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    sample_labels = model.fit_predict(sample_data)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(sample_data, sample_labels)
    labels = knn.predict(data)
    
    n_clusters = len(np.unique(labels))
    print(f"  检测到 {n_clusters} 个类别")
    
    return labels, model

def apply_birch(data, n_clusters, use_sampling):
    """Birch聚类"""
    model = Birch(
        n_clusters=n_clusters,
        threshold=0.5,
        branching_factor=50
    )
    
    labels = model.fit_predict(data)
    return labels, model

def apply_isodata(data, n_clusters, use_sampling):
    """ISODATA算法"""
    print("  使用ISODATA算法 (遥感经典方法)")
    
    sample_data, sample_indices = sample_data_for_training(data, 30000) if use_sampling else (data, None)
    
    max_clusters = n_clusters + 2
    min_clusters = max(2, n_clusters - 2)
    max_iterations = 20
    max_std = 1.0
    min_distance = 0.5
    
    current_centers = KMeans(n_clusters=n_clusters, random_state=42, n_init=3).fit(sample_data).cluster_centers_
    
    for iteration in range(max_iterations):
        from scipy.spatial.distance import cdist
        distances = cdist(sample_data, current_centers)
        labels = np.argmin(distances, axis=1)
        
        new_centers = []
        cluster_stds = []
        
        for i in range(len(current_centers)):
            mask = labels == i
            if np.sum(mask) == 0:
                continue
            
            cluster_data = sample_data[mask]
            center = np.mean(cluster_data, axis=0)
            std = np.std(cluster_data)
            
            new_centers.append(center)
            cluster_stds.append(std)
        
        new_centers = np.array(new_centers)
        
        if len(new_centers) < max_clusters:
            for i, std in enumerate(cluster_stds):
                if std > max_std and len(new_centers) < max_clusters:
                    offset = np.random.randn(new_centers.shape[1]) * std * 0.5
                    new_centers = np.vstack([new_centers, new_centers[i] + offset])
        
        if len(new_centers) > min_clusters:
            distances_centers = cdist(new_centers, new_centers)
            np.fill_diagonal(distances_centers, np.inf)
            min_dist_idx = np.unravel_index(np.argmin(distances_centers), distances_centers.shape)
            
            if distances_centers[min_dist_idx] < min_distance:
                merged_center = (new_centers[min_dist_idx[0]] + new_centers[min_dist_idx[1]]) / 2
                new_centers = np.delete(new_centers, min_dist_idx, axis=0)
                new_centers = np.vstack([new_centers, merged_center])
        
        if len(current_centers) == len(new_centers):
            if np.allclose(current_centers, new_centers, rtol=0.01):
                print(f"  ISODATA收敛于第 {iteration+1} 次迭代")
                break
        
        current_centers = new_centers
    
    if sample_indices is not None:
        from scipy.spatial.distance import cdist
        distances_full = cdist(data, current_centers)
        labels = np.argmin(distances_full, axis=1)
    else:
        from scipy.spatial.distance import cdist
        distances = cdist(sample_data, current_centers)
        labels = np.argmin(distances, axis=1)
    
    actual_clusters = len(current_centers)
    print(f"  ISODATA最终类别数: {actual_clusters}")
    
    class ISODATAModel:
        def __init__(self, centers):
            self.cluster_centers_ = centers
    
    return labels, ISODATAModel(current_centers)

def apply_hierarchical(data, n_clusters, use_sampling):
    """层次聚类"""
    sample_data, sample_indices = sample_data_for_training(data, 5000) if use_sampling else (data, None)
    
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    
    if sample_indices is not None:
        sample_labels = model.fit_predict(sample_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        centers = []
        for i in range(n_clusters):
            mask = sample_labels == i
            if np.sum(mask) > 0:
                centers.append(np.mean(sample_data[mask], axis=0))
        
        kmeans.cluster_centers_ = np.array(centers)
        labels = kmeans.predict(data)
        return labels, kmeans
    else:
        labels = model.fit_predict(data)
        return labels, model

def apply_classification_method(data_clean, n_clusters, method='minibatch_kmeans', 
                               random_state=42, use_sampling=True):
    """统一的分类方法接口"""
    n_samples = len(data_clean)
    print(f"数据: {n_samples:,}点, 特征: {data_clean.shape[1]}维, 目标: {n_clusters}类")
    
    np.random.seed(random_state)
    
    try:
        if method == 'kmeans':
            labels, model = apply_kmeans(data_clean, n_clusters, use_sampling)
        elif method == 'minibatch_kmeans':
            labels, model = apply_minibatch_kmeans(data_clean, n_clusters, use_sampling)
        elif method == 'gmm':
            labels, model = apply_gmm(data_clean, n_clusters, use_sampling)
        elif method == 'fuzzy_cmeans':
            labels, model = apply_fuzzy_cmeans(data_clean, n_clusters, use_sampling)
        elif method == 'spectral':
            labels, model = apply_spectral(data_clean, n_clusters, use_sampling)
        elif method == 'dbscan':
            labels, model = apply_dbscan(data_clean, use_sampling)
        elif method == 'optics':
            labels, model = apply_optics(data_clean, use_sampling)
        elif method == 'meanshift':
            labels, model = apply_meanshift(data_clean, use_sampling)
        elif method == 'birch':
            labels, model = apply_birch(data_clean, n_clusters, use_sampling)
        elif method == 'isodata':
            labels, model = apply_isodata(data_clean, n_clusters, use_sampling)
        elif method == 'hierarchical':
            labels, model = apply_hierarchical(data_clean, n_clusters, use_sampling)
        else:
            raise ValueError(f"未知方法: {method}")
        
        if labels.min() == 0:
            labels = labels + 1
        
        actual_n_clusters = len(np.unique(labels))
        print(f"✓ 分类完成! 实际类别: {actual_n_clusters}")
        
        return labels, model, actual_n_clusters
        
    except Exception as e:
        print(f"✗ 分类失败: {str(e)}")
        raise

def landsat8_unsupervised_classification(input_file, output_file, n_clusters=5,
                                        nodata_value=None, post_process=True,
                                        min_patch_size=50, smoothing_iterations=1,
                                        method='minibatch_kmeans', use_sampling=True):
    """主分类函数 - 使用优化的后处理"""
    print("="*80)
    print(" "*15 + "Landsat 8 非监督分类系统 (高性能版)")
    print("="*80)
    
    total_start = time.time()
    
    with PerformanceTimer("读取影像数据"):
        with rasterio.open(input_file) as src:
            bands_data = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
            
            if nodata_value is None and src.nodata is not None:
                nodata_value = src.nodata
            
            print(f"  形状: {bands_data.shape}, 波段: {src.count}, 尺寸: {src.width}×{src.height}")
    
    pixel_area = calculate_pixel_area(transform, crs)
    
    with PerformanceTimer("创建有效数据掩膜"):
        valid_mask_2d = create_valid_data_mask(bands_data, nodata_value)
    
    with PerformanceTimer("数据预处理"):
        height, width = bands_data.shape[1], bands_data.shape[2]
        n_bands = bands_data.shape[0]
        
        data_2d = bands_data.reshape(n_bands, -1).T
        valid_mask_1d = valid_mask_2d.ravel()
        data_valid = data_2d[valid_mask_1d]
        
        finite_mask = np.all(np.isfinite(data_valid), axis=1)
        data_clean_temp = data_valid[finite_mask]
        
        print(f"  有效数据: {len(data_clean_temp):,}点")
        
        scaler = StandardScaler()
        data_clean = scaler.fit_transform(data_clean_temp)
    
    with PerformanceTimer(f"执行分类 - {method}"):
        labels, model, actual_n_clusters = apply_classification_method(
            data_clean, n_clusters, method, use_sampling=use_sampling
        )
    
    with PerformanceTimer("重建分类结果"):
        full_labels_temp = np.zeros(len(data_valid), dtype=np.uint8)
        full_labels_temp[finite_mask] = labels
        
        full_labels = np.zeros(height * width, dtype=np.uint8)
        full_labels[valid_mask_1d] = full_labels_temp
        
        classification_result = full_labels.reshape(height, width)
    
    # 使用优化的后处理
    if post_process:
        classification_result = post_process_classification_fast(
            classification_result, valid_mask_2d, min_patch_size, smoothing_iterations
        )
    
    with PerformanceTimer("保存结果"):
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(classification_result, 1)
    
    area_stats = calculate_class_areas_enhanced(classification_result, valid_mask_2d, pixel_area, crs)
    
    total_time = time.time() - total_start
    print("\n" + "="*80)
    print(f"✓ 完成! 总耗时: {total_time:.2f}秒")
    print("="*80)
    
    return classification_result, model, valid_mask_2d, area_stats, pixel_area

# [可视化函数保持不变]

def plot_classification_result(classification_result, valid_mask, title_suffix="",
                               save_path='classification_result.png', dpi=150):
    """可视化分类结果"""
    with PerformanceTimer("生成分类图"):
        plot_data = classification_result.astype(float)
        plot_data[~valid_mask] = np.nan
        
        unique_classes = np.unique(classification_result[valid_mask])
        unique_classes = unique_classes[unique_classes > 0]
        n_classes = len(unique_classes)
        
        if n_classes == 0:
            print("⚠️  警告: 没有有效的分类结果可视化")
            return
        
        colors = ['#228B22', '#32CD32', '#90EE90', '#FFD700', '#FFA500',
                  '#8B4513', '#4169E1', '#87CEEB', '#808080', '#DC143C',
                  '#9370DB', '#FF69B4', '#00CED1', '#FF6347', '#4682B4']
        
        if n_classes > len(colors):
            import matplotlib.cm as cm
            cmap_colors = cm.get_cmap('tab20', n_classes)
            colors = [cmap_colors(i) for i in range(n_classes)]
        
        cmap = ListedColormap(colors[:n_classes])
        
        plt.figure(figsize=(14, 10))
        im = plt.imshow(plot_data, cmap=cmap, interpolation='nearest')
        
        cbar = plt.colorbar(im, label='土地类别', shrink=0.7)
        cbar.set_ticks(unique_classes)
        cbar.set_ticklabels([f'类别 {int(i)}' for i in unique_classes])
        
        title = f'Landsat 8 分类结果 ({n_classes}类)'
        if title_suffix:
            title += f" - {title_suffix}"
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.xlabel('列', fontsize=11)
        plt.ylabel('行', fontsize=11)
        
        legend_elements = [
            Patch(facecolor=colors[i], label=f'类别 {int(unique_classes[i])}')
            for i in range(n_classes)
        ]
        plt.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(1.15, 1), title="类型", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  保存: {save_path}")
        plt.close()

def plot_area_distribution(area_stats, method_name, save_path='面积分布.png', dpi=150):
    """面积分布图"""
    if not area_stats or len(area_stats) == 0:
        print("⚠️  警告: 没有面积统计数据可绘制")
        return
    
    with PerformanceTimer("生成面积图"):
        classes = []
        areas_km2 = []
        areas_ha = []
        percentages = []
        
        for stat in area_stats:
            area_km2 = stat['面积_平方千米']
            area_ha = stat['面积_公顷']
            pct = stat['占比_百分比']
            
            if np.isfinite(area_km2) and np.isfinite(pct) and area_km2 > 0:
                classes.append(f'类别{stat["类别"]}')
                areas_km2.append(float(area_km2))
                areas_ha.append(float(area_ha))
                percentages.append(float(pct))
        
        if len(areas_km2) == 0:
            print("⚠️  警告: 没有有效的面积数据可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        colors_array = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        # 柱状图 - 平方千米
        bars1 = ax1.bar(classes, areas_km2, color=colors_array, edgecolor='black')
        ax1.set_title(f'{method_name} - 面积分布 (km²)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('类别', fontsize=10)
        ax1.set_ylabel('面积 (km²)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, area, pct in zip(bars1, areas_km2, percentages):
            height = bar.get_height()
            if np.isfinite(height) and height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{area:.2f}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=8)
        
        # 柱状图 - 公顷
        bars2 = ax2.bar(classes, areas_ha, color=colors_array, edgecolor='black')
        ax2.set_title(f'{method_name} - 面积分布 (公顷)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('类别', fontsize=10)
        ax2.set_ylabel('面积 (公顷)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, area, pct in zip(bars2, areas_ha, percentages):
            height = bar.get_height()
            if np.isfinite(height) and height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{area:.1f}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontsize=8)
        
        # 饼图
        try:
            wedges, texts, autotexts = ax3.pie(
                areas_km2, labels=classes, autopct='%1.1f%%',
                colors=colors_array, startangle=90
            )
            ax3.set_title(f'{method_name} - 面积占比', fontsize=12, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        except Exception as e:
            ax3.text(0.5, 0.5, '绘图失败', ha='center', va='center', fontsize=14)
        
        # 表格
        table_data = []
        for cls, km2, ha, pct in zip(classes, areas_km2, areas_ha, percentages):
            table_data.append([cls, f'{km2:.4f}', f'{ha:.2f}', f'{pct:.2f}%'])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(
            cellText=table_data,
            colLabels=['类别', '面积(km²)', '面积(公顷)', '占比(%)'],
            cellLoc='center',
            loc='center',
            colColours=['lightgray']*4
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title('详细数据', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"  保存: {save_path}")
        plt.close()

def save_area_statistics(area_stats, output_file="面积统计.csv"):
    """保存统计"""
    if not area_stats or len(area_stats) == 0:
        print("⚠️  警告: 没有统计数据可保存")
        return None
    
    try:
        df = pd.DataFrame(area_stats)
        df = df.dropna()
        
        if df.empty:
            print("⚠️  警告: 清理后统计数据为空")
            return None
        
        df = df.sort_values('面积_平方千米', ascending=False)
        df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"  保存: {output_file}")
        
        try:
            excel_file = output_file.replace('.csv', '.xlsx')
            df.to_excel(excel_file, index=False, float_format='%.6f')
            print(f"  保存: {excel_file}")
        except:
            pass
        
        return df
    except Exception as e:
        print(f"⚠️  警告: 保存统计数据失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*10 + "Landsat 8 非监督分类系统 v3.0 (高性能版)")
    print("="*80 + "\n")
    
    input_file = r"D:\code313\Geo_programe\rasterio\data\XZ_SQ_L8_2024.tif"
    
    if not os.path.exists(input_file):
        print(f"✗ 文件不存在: {input_file}")
        return
    
    print_methods_table()
    
    methods = list_classification_methods()
    available_methods = {k: v for k, v in methods.items() if v['available']}
    
    method_input = input("\n请选择方法 (输入序号或名称，默认minibatch_kmeans): ").strip()
    
    if not method_input:
        method = 'minibatch_kmeans'
    elif method_input.isdigit():
        available_keys = list(available_methods.keys())
        idx = int(method_input) - 1
        if 0 <= idx < len(available_keys):
            method = available_keys[idx]
        else:
            method = 'minibatch_kmeans'
    elif method_input.lower() in available_methods:
        method = method_input.lower()
    else:
        method = 'minibatch_kmeans'
    
    if method not in available_methods:
        print(f"⚠️  方法 {method} 不可用，使用默认方法")
        method = 'minibatch_kmeans'
    
    method_info = methods[method]
    print(f"\n✓ 选择: {method_info['name']}")
    
    if method_info['need_n_clusters']:
        n_input = input(f"\n分类数量 (2-15，默认6): ").strip()
        try:
            n_clusters = int(n_input) if n_input else 6
            n_clusters = max(2, min(15, n_clusters))
        except:
            n_clusters = 6
    else:
        n_clusters = 6
    
    print(f"✓ 目标类别: {n_clusters}")
    
    post_input = input("\n后处理? (y/n，默认y): ").strip().lower()
    post_process = post_input != 'n'
    
    if post_process:
        min_patch_size = 300
        smoothing_iterations = 1
        print(f"✓ 后处理: 最小斑块={min_patch_size}, 平滑={smoothing_iterations}")
    else:
        min_patch_size = 0
        smoothing_iterations = 0
    
    sampling_input = input("\n大数据采样加速? (y/n，默认y): ").strip().lower()
    use_sampling = sampling_input != 'n'
    print(f"✓ 采样: {'是' if use_sampling else '否'}")
    
    output_file = f"classification_{method}_{n_clusters}classes.tif"
    
    try:
        classification_result, model, valid_mask, area_stats, pixel_area = \
            landsat8_unsupervised_classification(
                input_file, output_file, n_clusters,
                post_process=post_process,
                min_patch_size=min_patch_size,
                smoothing_iterations=smoothing_iterations,
                method=method,
                use_sampling=use_sampling
            )
        
        csv_file = f"统计_{method}.csv"
        df = save_area_statistics(area_stats, csv_file)
        
        plot_file = f"分类_{method}.png"
        plot_classification_result(classification_result, valid_mask, 
                                  method_info['name'], plot_file)
        
        area_file = f"面积_{method}.png"
        plot_area_distribution(area_stats, method_info['name'], area_file)
        
        print("\n" + "="*80)
        print("✓ 全部完成!")
        print("="*80)
        print("输出文件:")
        print(f"  1. {output_file}")
        if df is not None:
            print(f"  2. {csv_file}")
        print(f"  3. {plot_file}")
        print(f"  4. {area_file}")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ 处理失败!")
        print("="*80)
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()