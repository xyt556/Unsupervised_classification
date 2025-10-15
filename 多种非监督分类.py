import numpy as np
import rasterio
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.ndimage import label, binary_dilation
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于显示负号

def get_current_paths():
    """获取当前工作目录和脚本目录"""
    current_working_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前工作目录: {current_working_dir}")
    print(f"脚本所在目录: {script_dir}")
    return current_working_dir, script_dir

def list_classification_methods():
    """列出可用的分类方法"""
    methods = {
        'kmeans': 'K-Means聚类 (默认)',
        'minibatch_kmeans': 'MiniBatch K-Means (适合大数据集)',
        'gmm': '高斯混合模型 (GMM)',
        'hierarchical': '层次聚类',
        'dbscan': 'DBSCAN (基于密度)'
    }
    return methods

def create_valid_data_mask(bands_data, nodata_value=None):
    """
    创建有效数据掩膜，排除无数据区域
    """
    # 如果没有指定nodata值，使用默认方法检测
    if nodata_value is None:
        # 方法1: 检测全为0的像素（常见于背景）
        zero_mask = np.all(bands_data == 0, axis=0)
        
        # 方法2: 检测NaN值
        nan_mask = np.any(np.isnan(bands_data), axis=0)
        
        # 方法3: 检测异常低值（假设小于-100为无效）
        low_value_mask = np.any(bands_data < -100, axis=0)
        
        # 合并所有无效区域
        invalid_mask = zero_mask | nan_mask | low_value_mask
    else:
        # 使用指定的nodata值
        invalid_mask = np.any(bands_data == nodata_value, axis=0)
    
    # 有效数据掩膜是无效掩膜的反转
    valid_mask = ~invalid_mask
    
    print(f"有效像素比例: {np.sum(valid_mask) / valid_mask.size * 100:.2f}%")
    return valid_mask

def apply_classification_method(data_clean, n_clusters, method='kmeans', random_state=42):
    """
    应用指定的分类方法
    """
    print(f"使用分类方法: {method}")
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(data_clean)
        
    elif method == 'minibatch_kmeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1000)
        labels = model.fit_predict(data_clean)
        
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(data_clean)
        
    elif method == 'hierarchical':
        # 对于大数据集，使用小批量层次聚类
        if len(data_clean) > 10000:
            # 采样处理大数据
            indices = np.random.choice(len(data_clean), min(10000, len(data_clean)), replace=False)
            sample_data = data_clean[indices]
            model = AgglomerativeClustering(n_clusters=n_clusters)
            sample_labels = model.fit_predict(sample_data)
            
            # 使用KMeans将结果扩展到整个数据集
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(sample_data)
            _, indices = nn.kneighbors(data_clean)
            labels = sample_labels[indices.flatten()]
            model = None  # 层次聚类模型不适合直接返回
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(data_clean)
            
    elif method == 'dbscan':
        # DBSCAN不需要指定聚类数量，但需要调整参数
        model = DBSCAN(eps=0.5, min_samples=10)
        labels = model.fit_predict(data_clean)
        # 将-1（噪声）重新标记为0，并调整标签从1开始
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels, 1)}
        labels = np.array([label_mapping[l] for l in labels])
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"DBSCAN自动检测到 {n_clusters} 个类别")
        
    else:
        raise ValueError(f"未知的分类方法: {method}")
    
    return labels, model, n_clusters

def post_process_classification(classification_result, valid_mask, min_patch_size=50, smoothing_iterations=1):
    """
    分类后处理：去除小斑块和平滑分类结果
    """
    print("开始分类后处理...")
    
    # 创建处理结果的副本
    processed_result = classification_result.copy()
    
    # 只对有效区域进行处理
    valid_area = valid_mask & (classification_result > 0)
    
    # 统计原始分类的类别数量
    unique_classes = np.unique(classification_result[valid_area])
    print(f"后处理前类别数量: {len(unique_classes)}")
    
    # 步骤1: 去除小斑块
    if min_patch_size > 0:
        print(f"去除小斑块 (最小尺寸: {min_patch_size} 像素)...")
        processed_result = remove_small_patches(processed_result, valid_area, min_patch_size)
    
    # 步骤2: 应用多数滤波平滑
    if smoothing_iterations > 0:
        print(f"应用多数滤波平滑 (迭代次数: {smoothing_iterations})...")
        for i in range(smoothing_iterations):
            processed_result = majority_filter_smoothing(processed_result, valid_area)
            print(f"  平滑迭代 {i+1} 完成")
    
    # 统计后处理后的类别数量
    unique_classes_after = np.unique(processed_result[valid_area])
    print(f"后处理后类别数量: {len(unique_classes_after)}")
    
    print("分类后处理完成!")
    return processed_result

def remove_small_patches(classification_result, valid_mask, min_patch_size):
    """
    移除小斑块 - 修复版本
    """
    processed_result = classification_result.copy()
    
    # 获取所有类别（排除0，即无数据）
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    print(f"处理 {len(unique_classes)} 个类别的小斑块...")
    
    # 为整个图像创建标记数组
    full_labeled, num_features = label(valid_mask)
    
    # 计算每个连通区域的大小和主要类别
    component_sizes = []
    component_classes = []
    
    for i in range(1, num_features + 1):
        component_mask = full_labeled == i
        component_pixels = classification_result[component_mask]
        
        # 找出该区域的主要类别
        if len(component_pixels) > 0:
            unique, counts = np.unique(component_pixels, return_counts=True)
            main_class = unique[np.argmax(counts)]
            component_sizes.append(np.sum(component_mask))
            component_classes.append(main_class)
        else:
            component_sizes.append(0)
            component_classes.append(0)
    
    # 找出小斑块并重新分类
    small_patch_count = 0
    for i in range(1, num_features + 1):
        if component_sizes[i-1] < min_patch_size and component_classes[i-1] > 0:
            component_mask = full_labeled == i
            main_class = component_classes[i-1]
            
            # 找到相邻像素
            dilated_mask = binary_dilation(component_mask, structure=np.ones((3,3)))
            neighbor_mask = dilated_mask & ~component_mask & valid_mask
            
            if np.any(neighbor_mask):
                # 找到相邻像素中最常见的类别
                neighbor_classes = classification_result[neighbor_mask]
                unique, counts = np.unique(neighbor_classes, return_counts=True)
                if len(unique) > 0:
                    new_class = unique[np.argmax(counts)]
                    processed_result[component_mask] = new_class
                    small_patch_count += 1
    
    print(f"移除了 {small_patch_count} 个小斑块")
    return processed_result

def majority_filter_smoothing(classification_result, valid_mask):
    """
    使用多数滤波平滑分类结果 - 修复版本
    """
    from scipy.ndimage import generic_filter
    
    # 定义多数函数
    def majority_func(values):
        # 忽略0值（无数据）和无效值
        non_zero = values[values > 0]
        if len(non_zero) == 0:
            return 0
        # 返回出现次数最多的值
        values, counts = np.unique(non_zero, return_counts=True)
        return values[np.argmax(counts)]
    
    # 创建处理结果的副本
    smoothed_result = classification_result.copy()
    
    # 只对有效区域进行处理
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) > 0:
        # 使用3x3窗口进行多数滤波
        footprint = np.ones((3, 3))
        
        # 对整幅图像应用滤波
        temp_result = generic_filter(
            classification_result, 
            majority_func, 
            footprint=footprint,
            mode='constant', 
            cval=0
        )
        
        # 只更新有效区域的像素
        smoothed_result[valid_mask] = temp_result[valid_mask]
    
    return smoothed_result

def calculate_pixel_area(transform):
    """
    计算单个像素的面积（平方米）
    """
    # 获取像素的宽度和高度（单位：米）
    pixel_width = transform[0]
    pixel_height = -transform[4]  # 因为是北向上，所以高度是负值
    
    # 计算像素面积（平方米）
    pixel_area = pixel_width * pixel_height
    
    print(f"像素尺寸: {pixel_width:.2f} x {pixel_height:.2f} 米")
    print(f"单个像素面积: {pixel_area:.2f} 平方米")
    
    return pixel_area

def calculate_class_areas(classification_result, valid_mask, pixel_area, n_clusters):
    """
    计算每个类别的面积（平方米和平方千米）
    """
    print("\n=== 各类别面积统计 ===")
    
    # 只统计有效区域内的像素
    total_valid_pixels = np.sum(valid_mask)
    total_area_m2 = total_valid_pixels * pixel_area
    total_area_km2 = total_area_m2 / 1e6
    
    print(f"总有效面积: {total_area_km2:.2f} 平方千米")
    
    # 获取实际存在的类别
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    area_stats = []
    
    for cluster_id in unique_classes:
        mask = (classification_result == cluster_id) & valid_mask
        cluster_pixels = np.sum(mask)
        
        if cluster_pixels == 0:
            continue
            
        # 计算面积
        area_m2 = cluster_pixels * pixel_area
        area_km2 = area_m2 / 1e6
        percentage = cluster_pixels / total_valid_pixels * 100
        
        print(f"\n类别 {cluster_id}:")
        print(f"  像素数量: {cluster_pixels}")
        print(f"  面积: {area_km2:.2f} 平方千米 ({area_m2:.0f} 平方米)")
        print(f"  占比: {percentage:.2f}%")
        
        area_stats.append({
            '类别': cluster_id,
            '像素数量': cluster_pixels,
            '面积_平方千米': area_km2,
            '面积_平方米': area_m2,
            '占比_百分比': percentage
        })
    
    return area_stats

def save_area_statistics(area_stats, output_file="面积统计.csv"):
    """
    保存面积统计结果到CSV文件
    """
    df = pd.DataFrame(area_stats)
    df = df.sort_values('面积_平方千米', ascending=False)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n面积统计结果已保存至: {output_file}")
    return df

def landsat8_unsupervised_classification(input_file, output_file, n_clusters=5, nodata_value=None, 
                                        post_process=True, min_patch_size=50, smoothing_iterations=1,
                                        method='kmeans'):
    """
    Landsat 8非监督分类（土地利用分类）
    
    参数:
    input_file: 输入的Landsat 8多波段文件
    output_file: 输出分类结果文件
    n_clusters: 分类数量，默认为5类
    nodata_value: 无数据值，如果为None则自动检测
    post_process: 是否进行后处理
    min_patch_size: 后处理中的最小斑块大小
    smoothing_iterations: 后处理中的平滑迭代次数
    method: 分类方法 ('kmeans', 'minibatch_kmeans', 'gmm', 'hierarchical', 'dbscan')
    """
    
    # 读取Landsat 8数据
    with rasterio.open(input_file) as src:
        # 读取所有波段数据
        bands_data = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
        
        # 如果未指定nodata值，尝试从源文件中获取
        if nodata_value is None and src.nodata is not None:
            nodata_value = src.nodata
        
        print(f"数据形状: {bands_data.shape}")
        print(f"波段数: {src.count}")
        print(f"影像尺寸: {src.width} x {src.height}")
        print(f"无数据值: {nodata_value}")
        print(f"坐标参考系统: {crs}")
    
    # 计算像素面积
    pixel_area = calculate_pixel_area(transform)
    
    # 创建有效数据掩膜
    valid_mask_2d = create_valid_data_mask(bands_data, nodata_value)
    
    # 重新组织数据格式
    height, width = bands_data.shape[1], bands_data.shape[2]
    n_bands = bands_data.shape[0]
    
    # 重塑数据为二维数组
    data_2d = bands_data.reshape(n_bands, -1).T  # 转置为 (像素数, 波段数)
    
    print(f"重塑后数据形状: {data_2d.shape}")
    
    # 数据预处理 - 标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_2d)
    
    # 移除无效值（NaN、无穷大和无数据区域）
    valid_mask_1d = valid_mask_2d.reshape(-1)
    computational_mask = ~np.any(np.isnan(data_scaled) | np.isinf(data_scaled), axis=1)
    final_valid_mask = valid_mask_1d & computational_mask
    
    data_clean = data_scaled[final_valid_mask]
    
    print(f"有效像素数: {len(data_clean)}")
    print(f"最终有效像素比例: {len(data_clean) / len(data_2d) * 100:.2f}%")
    
    # 执行指定的分类方法
    print(f"正在进行{method}聚类，类别数: {n_clusters}...")
    labels, model, actual_n_clusters = apply_classification_method(data_clean, n_clusters, method)
    
    # 创建完整的分类结果数组（0表示无数据区域）
    full_labels = np.zeros(height * width, dtype=np.uint8)
    full_labels[final_valid_mask] = labels
    
    # 重塑为原始影像形状
    classification_result = full_labels.reshape(height, width)
    
    # 应用后处理
    if post_process:
        classification_result = post_process_classification(
            classification_result, valid_mask_2d, min_patch_size, smoothing_iterations
        )
    else:
        print("跳过分类后处理")
    
    # 更新输出文件的元数据
    profile.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'compress': 'lzw',
        'nodata': 0  # 设置0为无数据值
    })
    
    # 保存分类结果
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(classification_result, 1)
        dst.set_band_description(1, f"土地利用分类 ({actual_n_clusters}类)")
    
    print(f"分类完成！结果已保存至: {output_file}")
    
    # 计算各类别面积
    area_stats = calculate_class_areas(classification_result, valid_mask_2d, pixel_area, actual_n_clusters)
    
    return classification_result, model, valid_mask_2d, area_stats, pixel_area

def plot_classification_result(classification_result, valid_mask, n_clusters, title_suffix="", save_path='classification_result.png'):
    """可视化分类结果，无数据区域不显示"""
    
    # 创建分类结果的副本，无数据区域设置为NaN以便在图中显示为透明
    plot_data = classification_result.astype(float)
    plot_data[~valid_mask] = np.nan
    
    # 统计实际存在的类别
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]  # 排除0（无数据）
    actual_clusters = len(unique_classes)
    
    print(f"可视化类别数量: {actual_clusters}")
    
    # 创建自定义颜色映射
    colors = ['darkgreen', 'green', 'lightgreen', 'yellow', 'orange', 
              'brown', 'blue', 'lightblue', 'gray', 'red', 'purple', 'pink']
    
    # 确保颜色数量足够
    if actual_clusters > len(colors):
        # 如果类别太多，生成更多颜色
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab20', actual_clusters)
        colors = [color_map(i) for i in range(actual_clusters)]
    
    cmap = ListedColormap(colors[:actual_clusters])
    
    plt.figure(figsize=(14, 10))
    
    # 显示分类结果，无数据区域自动透明
    im = plt.imshow(plot_data, cmap=cmap, vmin=0.5, vmax=actual_clusters+0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(im, label='土地利用类别', shrink=0.8)
    cbar.set_ticks(range(1, actual_clusters+1))
    cbar.set_ticklabels([f'类别 {i}' for i in range(1, actual_clusters+1)])
    
    title = f'Landsat 8土地利用分类结果 ({actual_clusters}类)'
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('像素列')
    plt.ylabel('像素行')
    
    # 添加图例
    legend_elements = [Patch(facecolor=colors[i], 
                           label=f'类别 {unique_classes[i]}') for i in range(actual_clusters)]
    plt.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(1.15, 1), title="土地类型", title_fontsize=12)
    
    # 添加有效区域信息
    valid_percentage = np.sum(valid_mask) / valid_mask.size * 100
    plt.figtext(0.02, 0.02, f'有效区域: {valid_percentage:.1f}%', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def analyze_cluster_statistics(original_data, classification_result, valid_mask, n_clusters):
    """分析各类别的统计特征"""
    
    print("\n=== 各类别统计特征 ===")
    
    band_names = ['蓝光', '绿光', '红光', '近红外', '短波红外1', '短波红外2', '热红外']
    
    # 只统计有效区域内的像素
    total_valid_pixels = np.sum(valid_mask)
    
    # 获取实际存在的类别
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    print(f"实际存在的类别数量: {len(unique_classes)}")
    
    for cluster_id in unique_classes:
        mask = (classification_result == cluster_id) & valid_mask
        cluster_pixels = np.sum(mask)
        
        if cluster_pixels == 0:
            print(f"\n类别 {cluster_id}: 无有效像素")
            continue
            
        print(f"\n类别 {cluster_id}:")
        print(f"  像素数量: {cluster_pixels} ({cluster_pixels/total_valid_pixels*100:.2f}%)")
        
        # 计算每个波段的均值
        band_means = []
        for band_idx, band_name in enumerate(band_names):
            if band_idx < original_data.shape[0]:
                band_mean = np.mean(original_data[band_idx][mask])
                band_means.append(band_mean)
                print(f"  {band_name}波段均值: {band_mean:.2f}")
        
        # 简单的地物类型推断（基于光谱特征）
        land_cover_type = infer_land_cover_type(band_means, band_names)
        print(f"  可能的地物类型: {land_cover_type}")

def infer_land_cover_type(band_means, band_names):
    """
    基于光谱特征简单推断地物类型
    这是一个简化的推断方法，实际应用中需要更复杂的规则
    """
    if len(band_means) < 4:
        return "未知"
    
    # 提取关键波段信息
    red = band_means[2] if len(band_means) > 2 else 0
    nir = band_means[3] if len(band_means) > 3 else 0
    swir1 = band_means[4] if len(band_means) > 4 else 0
    
    # 计算NDVI（归一化植被指数）
    if nir + red > 0:
        ndvi = (nir - red) / (nir + red)
    else:
        ndvi = 0
    
    # 简单分类规则
    if ndvi > 0.3:
        return "植被"
    elif ndvi > 0.1:
        return "稀疏植被"
    elif swir1 > 1000:  # 这个阈值需要根据实际数据调整
        return "裸地/建筑"
    elif nir < red and nir < band_means[0]:  # 近红外值较低
        return "水体"
    else:
        return "其他"

def compare_pre_post_processing(original_result, processed_result, valid_mask, n_clusters):
    """比较后处理前后的结果"""
    
    # 统计实际类别数量
    unique_original = np.unique(original_result[valid_mask])
    unique_original = unique_original[unique_original > 0]
    unique_processed = np.unique(processed_result[valid_mask])
    unique_processed = unique_processed[unique_processed > 0]
    
    print(f"原始分类类别数: {len(unique_original)}")
    print(f"后处理分类类别数: {len(unique_processed)}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 准备绘图数据
    plot_original = original_result.astype(float)
    plot_processed = processed_result.astype(float)
    plot_original[~valid_mask] = np.nan
    plot_processed[~valid_mask] = np.nan
    
    # 使用相同的颜色映射
    colors = ['darkgreen', 'green', 'lightgreen', 'yellow', 'orange', 
              'brown', 'blue', 'lightblue', 'gray', 'red', 'purple', 'pink']
    
    # 确定最大类别数量
    max_classes = max(len(unique_original), len(unique_processed))
    if max_classes > len(colors):
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab20', max_classes)
        colors = [color_map(i) for i in range(max_classes)]
    
    cmap_original = ListedColormap(colors[:len(unique_original)])
    cmap_processed = ListedColormap(colors[:len(unique_processed)])
    
    # 绘制原始结果
    im1 = ax1.imshow(plot_original, cmap=cmap_original, vmin=0.5, vmax=len(unique_original)+0.5)
    ax1.set_title(f'原始分类结果 ({len(unique_original)}类)', fontsize=14)
    ax1.set_xlabel('像素列')
    ax1.set_ylabel('像素行')
    
    # 绘制后处理结果
    im2 = ax2.imshow(plot_processed, cmap=cmap_processed, vmin=0.5, vmax=len(unique_processed)+0.5)
    ax2.set_title(f'后处理结果 ({len(unique_processed)}类)', fontsize=14)
    ax2.set_xlabel('像素列')
    ax2.set_ylabel('像素行')
    
    plt.tight_layout()
    plt.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_area_distribution(area_stats, method_name, save_path='面积分布图.png'):
    """绘制各类别面积分布图"""
    
    if not area_stats:
        print("没有面积统计数据可绘制")
        return
    
    # 准备数据
    classes = [f'类别 {stat["类别"]}' for stat in area_stats]
    areas = [stat['面积_平方千米'] for stat in area_stats]
    percentages = [stat['占比_百分比'] for stat in area_stats]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    bars = ax1.bar(classes, areas, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
    ax1.set_title(f'{method_name} - 各类别面积分布', fontsize=14)
    ax1.set_xlabel('土地类别')
    ax1.set_ylabel('面积 (平方千米)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱子上添加数值标签
    for bar, area, percentage in zip(bars, areas, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(areas)*0.01,
                f'{area:.1f} km²\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    wedges, texts, autotexts = ax2.pie(areas, labels=classes, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax2.set_title(f'{method_name} - 各类别面积占比', fontsize=14)
    
    # 美化饼图文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 配置参数
    get_current_paths()
    input_file = r"D:\code313\Geo_programe\rasterio\data\XZ_SQ_L8_2024.tif"
    output_file = "land_use_classification.tif"
    n_clusters = 6  # 根据您的需求调整分类数量
    
    # 显示可用的分类方法
    methods = list_classification_methods()
    print("\n可用的分类方法:")
    for key, desc in methods.items():
        print(f"  {key}: {desc}")
    
    # 让用户选择分类方法
    method = input("\n请选择分类方法 (默认: kmeans): ").strip().lower()
    if not method or method not in methods:
        method = 'kmeans'
        print(f"使用默认方法: {methods[method]}")
    else:
        print(f"选择的方法: {methods[method]}")
    
    # 后处理参数 - 使用更保守的设置
    post_process = True  # 是否进行后处理
    min_patch_size = 50  # 最小斑块大小（像素数）- 减小这个值
    smoothing_iterations = 1  # 平滑迭代次数 - 减少迭代次数
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        return
    
    try:
        # 执行非监督分类（包含后处理）
        classification_result, model, valid_mask, area_stats, pixel_area = landsat8_unsupervised_classification(
            input_file, output_file, n_clusters,
            post_process=post_process,
            min_patch_size=min_patch_size,
            smoothing_iterations=smoothing_iterations,
            method=method
        )
        
        # 保存面积统计结果
        area_df = save_area_statistics(area_stats, f"面积统计_{method}.csv")
        
        # 绘制面积分布图
        plot_area_distribution(area_stats, methods[method], f"面积分布图_{method}.png")
        
        # 为了比较，我们也生成一个未后处理的版本
        if post_process:
            print("\n生成未后处理版本用于比较...")
            original_result, _, _, _, _ = landsat8_unsupervised_classification(
                input_file, "land_use_classification_original.tif", n_clusters,
                post_process=False,
                method=method
            )
            
            # 比较后处理前后结果
            compare_pre_post_processing(original_result, classification_result, valid_mask, n_clusters)
        
        # 读取原始数据用于统计分析
        with rasterio.open(input_file) as src:
            original_data = src.read()
        
        # 分析统计特征
        analyze_cluster_statistics(original_data, classification_result, valid_mask, n_clusters)
        
        # 可视化结果（无数据区域不显示）
        title_suffix = f"{methods[method]} - 后处理" if post_process else f"{methods[method]} - 原始"
        save_path = f"classification_result_{method}_processed.png" if post_process else f"classification_result_{method}_original.png"
        plot_classification_result(classification_result, valid_mask, n_clusters, title_suffix, save_path)
        
        print(f"\n分类完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"分类方法: {methods[method]}")
        print(f"分类数量: {n_clusters}")
        if post_process:
            print(f"后处理参数: 最小斑块大小={min_patch_size}, 平滑迭代次数={smoothing_iterations}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()