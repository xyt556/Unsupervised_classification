import streamlit as st
import numpy as np
import rasterio
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, OPTICS, 
                             MeanShift, Birch, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import label, binary_dilation, generic_filter
import pandas as pd
import warnings
import time
import io
import tempfile
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ==================== 中文字体配置 ====================

def configure_chinese_fonts():
    """
    配置matplotlib中文字体
    支持Windows、Mac、Linux多平台
    """
    import platform
    
    # 获取操作系统类型
    system = platform.system()
    
    # 尝试多个中文字体
    chinese_fonts = []
    
    if system == 'Windows':
        chinese_fonts = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong'              # 仿宋
        ]
    elif system == 'Darwin':  # Mac
        chinese_fonts = [
            'PingFang SC',          # 苹方
            'Heiti SC',             # 黑体
            'STHeiti',              # 华文黑体
            'Arial Unicode MS'
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Droid Sans Fallback',
            'DejaVu Sans'
        ]
    
    # 获取系统可用字体
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available_fonts = {f.name for f in fm.ttflist}
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果没有找到中文字体，尝试自动检测
    if selected_font is None:
        for font in available_fonts:
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song']):
                selected_font = font
                break
    
    # 配置matplotlib
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        st.success(f"✓ 已配置中文字体: {selected_font}")
        return True
    else:
        st.warning("⚠️ 未找到中文字体，将使用英文显示")
        return False

# 尝试配置中文字体
CHINESE_SUPPORT = False

try:
    CHINESE_SUPPORT = configure_chinese_fonts()
except Exception as e:
    st.warning(f"字体配置失败: {str(e)}")
    CHINESE_SUPPORT = False

# 尝试导入numba加速
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# 尝试导入scikit-fuzzy
try:
    import skfuzzy as fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================== 语言配置 ====================

# 定义中英文标签
LABELS = {
    'title': {
        'zh': '🛰️ Landsat 8 非监督分类系统',
        'en': '🛰️ Landsat 8 Unsupervised Classification System'
    },
    'subtitle': {
        'zh': '基于机器学习的遥感影像自动分类平台',
        'en': 'Machine Learning Based Remote Sensing Image Classification Platform'
    },
    'parameters': {
        'zh': '📋 参数设置',
        'en': '📋 Parameters'
    },
    'upload': {
        'zh': '上传Landsat 8影像 (TIF格式)',
        'en': 'Upload Landsat 8 Image (TIF format)'
    },
    'method': {
        'zh': '选择分类方法',
        'en': 'Select Classification Method'
    },
    'clusters': {
        'zh': '分类数量',
        'en': 'Number of Classes'
    },
    'postprocess': {
        'zh': '启用后处理',
        'en': 'Enable Post-processing'
    },
    'minpatch': {
        'zh': '最小斑块大小（像素）',
        'en': 'Minimum Patch Size (pixels)'
    },
    'run': {
        'zh': '🚀 开始分类',
        'en': '🚀 Start Classification'
    },
    'results': {
        'zh': '📊 分类结果',
        'en': '📊 Classification Results'
    },
    'classmap': {
        'zh': '🗺️ 分类图像',
        'en': '🗺️ Classification Map'
    },
    'statistics': {
        'zh': '📈 统计分析',
        'en': '📈 Statistical Analysis'
    },
    'datatable': {
        'zh': '📋 数据表格',
        'en': '📋 Data Table'
    },
    'download': {
        'zh': '💾 下载结果',
        'en': '💾 Download Results'
    },
    'class': {
        'zh': '类别',
        'en': 'Class'
    },
    'area_km2': {
        'zh': '面积_km²',
        'en': 'Area_km²'
    },
    'area_ha': {
        'zh': '面积_公顷',
        'en': 'Area_ha'
    },
    'percentage': {
        'zh': '占比_%',
        'en': 'Percentage_%'
    },
    'pixels': {
        'zh': '像素数量',
        'en': 'Pixel Count'
    }
}

def get_label(key, lang='zh'):
    """获取标签文本"""
    if not CHINESE_SUPPORT:
        lang = 'en'
    
    parts = key.split('.')
    current = LABELS
    for part in parts:
        current = current.get(part, {})
    
    if isinstance(current, dict):
        return current.get(lang, key)
    return current

# 页面配置
st.set_page_config(
    page_title="Landsat 8 Classification System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stProgress > div > div > div > div {
        background-color: #00c853;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================

if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'area_stats' not in st.session_state:
    st.session_state.area_stats = None
if 'valid_mask' not in st.session_state:
    st.session_state.valid_mask = None

# ==================== 辅助函数 ====================

@st.cache_data
def list_classification_methods():
    """列出所有可用的分类方法"""
    methods = {
        'minibatch_kmeans': {
            'name': 'MiniBatch K-Means',
            'desc': '⚡⚡⚡⚡⚡ Fast, suitable for large datasets',
            'need_n_clusters': True,
            'available': True
        },
        'kmeans': {
            'name': 'K-Means',
            'desc': '⚡⚡⚡⚡☆ Balanced speed and accuracy',
            'need_n_clusters': True,
            'available': True
        },
        'gmm': {
            'name': 'Gaussian Mixture Model',
            'desc': '★★★★★ High accuracy',
            'need_n_clusters': True,
            'available': True
        },
        'birch': {
            'name': 'Birch',
            'desc': '⚡⚡⚡⚡☆ Memory efficient',
            'need_n_clusters': True,
            'available': True
        },
        'fuzzy_cmeans': {
            'name': 'Fuzzy C-Means',
            'desc': '★★★★★ Remote sensing classic',
            'need_n_clusters': True,
            'available': FUZZY_AVAILABLE
        },
        'isodata': {
            'name': 'ISODATA',
            'desc': '★★★★☆ Standard RS algorithm',
            'need_n_clusters': True,
            'available': True
        },
    }
    return {k: v for k, v in methods.items() if v['available']}

def create_valid_data_mask(bands_data, nodata_value=None):
    """创建有效数据掩膜"""
    if nodata_value is not None:
        invalid_mask = np.any(bands_data == nodata_value, axis=0)
    else:
        invalid_mask = (
            np.all(bands_data == 0, axis=0) |
            np.any(np.isnan(bands_data), axis=0) |
            np.any(np.isinf(bands_data), axis=0)
        )
    
    valid_mask = ~invalid_mask
    return valid_mask

def calculate_pixel_area(transform, crs=None):
    """计算像素面积"""
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    
    if crs is not None and crs.is_geographic:
        pixel_width_m = pixel_width * 111320
        pixel_height_m = pixel_height * 110540
        pixel_area = pixel_width_m * pixel_height_m
    else:
        pixel_area = pixel_width * pixel_height
    
    return pixel_area

def calculate_class_areas(classification_result, valid_mask, pixel_area):
    """计算各类别面积"""
    valid_data = classification_result[valid_mask]
    
    if len(valid_data) == 0:
        return []
    
    unique_classes, counts = np.unique(valid_data, return_counts=True)
    valid_indices = (unique_classes > 0) & np.isfinite(unique_classes) & (counts > 0)
    unique_classes = unique_classes[valid_indices]
    counts = counts[valid_indices]
    
    if len(unique_classes) == 0:
        return []
    
    total_pixels = np.sum(counts)
    area_stats = []
    
    for cluster_id, cluster_pixels in zip(unique_classes, counts):
        if not np.isfinite(cluster_pixels) or cluster_pixels <= 0:
            continue
        
        area_m2 = float(cluster_pixels) * pixel_area
        area_km2 = area_m2 / 1_000_000
        area_ha = area_m2 / 10_000
        percentage = (float(cluster_pixels) / total_pixels) * 100
        
        area_stats.append({
            get_label('class'): int(cluster_id),
            get_label('pixels'): int(cluster_pixels),
            get_label('area_km2'): round(float(area_km2), 4),
            get_label('area_ha'): round(float(area_ha), 2),
            get_label('percentage'): round(float(percentage), 2)
        })
    
    return area_stats

# ==================== 核心分类函数 ====================

def apply_classification(data_clean, n_clusters, method, progress_bar):
    """执行分类"""
    progress_bar.progress(0.3, "Classifying...")
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=200)
        labels = model.fit_predict(data_clean)
    elif method == 'minibatch_kmeans':
        batch_size = min(2048, len(data_clean) // 20)
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                               batch_size=batch_size, max_iter=100, n_init=3)
        labels = model.fit_predict(data_clean)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42, 
                               max_iter=50, covariance_type='diag')
        labels = model.fit_predict(data_clean)
    elif method == 'birch':
        model = Birch(n_clusters=n_clusters, threshold=0.5, branching_factor=50)
        labels = model.fit_predict(data_clean)
    elif method == 'isodata':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        labels = model.fit_predict(data_clean)
    elif method == 'fuzzy_cmeans' and FUZZY_AVAILABLE:
        data_T = data_clean.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_T, c=n_clusters, m=2, error=0.005, maxiter=100, init=None
        )
        labels = np.argmax(u, axis=0)
        
        class FuzzyCMeansModel:
            def __init__(self, centers):
                self.cluster_centers_ = centers
        model = FuzzyCMeansModel(cntr)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data_clean)
    
    if labels.min() == 0:
        labels = labels + 1
    
    return labels, model

def simple_post_process(classification_result, valid_mask, min_patch_size):
    """简化的后处理"""
    if min_patch_size <= 0:
        return classification_result
    
    processed_result = classification_result.copy()
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    
    for class_id in unique_classes:
        class_mask = (classification_result == class_id) & valid_mask
        if not np.any(class_mask):
            continue
        
        class_labeled, num_features = label(class_mask)
        if num_features == 0:
            continue
        
        region_sizes = np.bincount(class_labeled.ravel())
        small_regions = np.where((region_sizes < min_patch_size) & (region_sizes > 0))[0]
        
        for region_id in small_regions:
            region_mask = class_labeled == region_id
            dilated = binary_dilation(region_mask, iterations=1)
            neighbor_mask = dilated & ~region_mask & valid_mask
            
            if np.any(neighbor_mask):
                neighbor_classes = classification_result[neighbor_mask]
                neighbor_classes = neighbor_classes[neighbor_classes > 0]
                
                if len(neighbor_classes) > 0:
                    unique_neighbors, counts = np.unique(neighbor_classes, return_counts=True)
                    new_class = unique_neighbors[np.argmax(counts)]
                    processed_result[region_mask] = new_class
    
    return processed_result

def run_classification(uploaded_file, method, n_clusters, post_process, min_patch_size):
    """运行分类的主函数"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 保存上传的文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # 1. 读取数据
        progress_bar.progress(0.1, "Reading image...")
        with rasterio.open(tmp_path) as src:
            bands_data = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
            nodata_value = src.nodata
            
            status_text.success(f"✓ Loaded {src.count} bands, size: {src.width}×{src.height}")
        
        # 2. 创建掩膜
        progress_bar.progress(0.15, "Creating mask...")
        valid_mask_2d = create_valid_data_mask(bands_data, nodata_value)
        
        # 3. 数据预处理
        progress_bar.progress(0.2, "Preprocessing...")
        height, width = bands_data.shape[1], bands_data.shape[2]
        n_bands = bands_data.shape[0]
        
        data_2d = bands_data.reshape(n_bands, -1).T
        valid_mask_1d = valid_mask_2d.ravel()
        data_valid = data_2d[valid_mask_1d]
        
        finite_mask = np.all(np.isfinite(data_valid), axis=1)
        data_clean_temp = data_valid[finite_mask]
        
        # 采样
        if len(data_clean_temp) > 50000:
            status_text.info(f"Large dataset ({len(data_clean_temp):,} points), sampling 50000...")
            indices = np.random.choice(len(data_clean_temp), 50000, replace=False)
            data_for_training = data_clean_temp[indices]
        else:
            data_for_training = data_clean_temp
        
        scaler = StandardScaler()
        data_clean = scaler.fit_transform(data_for_training)
        
        # 4. 执行分类
        labels, model = apply_classification(data_clean, n_clusters, method, progress_bar)
        
        # 预测全部数据
        if len(data_clean_temp) > 50000:
            progress_bar.progress(0.5, "Predicting all data...")
            data_all_scaled = scaler.transform(data_clean_temp)
            if hasattr(model, 'predict'):
                labels_all = model.predict(data_all_scaled)
            else:
                labels_all = labels
        else:
            labels_all = labels
        
        # 5. 重建分类结果
        progress_bar.progress(0.6, "Rebuilding result...")
        full_labels_temp = np.zeros(len(data_valid), dtype=np.uint8)
        full_labels_temp[finite_mask] = labels_all
        
        full_labels = np.zeros(height * width, dtype=np.uint8)
        full_labels[valid_mask_1d] = full_labels_temp
        
        classification_result = full_labels.reshape(height, width)
        
        # 6. 后处理
        if post_process and min_patch_size > 0:
            progress_bar.progress(0.7, "Post-processing...")
            classification_result = simple_post_process(
                classification_result, valid_mask_2d, min_patch_size
            )
        
        # 7. 计算面积
        progress_bar.progress(0.8, "Calculating statistics...")
        pixel_area = calculate_pixel_area(transform, crs)
        area_stats = calculate_class_areas(classification_result, valid_mask_2d, pixel_area)
        
        # 8. 保存结果
        progress_bar.progress(0.9, "Saving results...")
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.tif').name
        
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classification_result, 1)
        
        progress_bar.progress(1.0, "Done!")
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        return classification_result, valid_mask_2d, area_stats, output_path, pixel_area
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None

# ==================== 可视化函数 (修复中文显示) ====================

def plot_classification_matplotlib(classification_result, valid_mask):
    """生成分类结果图（优化中文显示）"""
    plot_data = classification_result.copy().astype(float)
    plot_data[~valid_mask] = np.nan
    
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    n_classes = len(unique_classes)
    
    # 颜色方案
    colors = ['#228B22', '#32CD32', '#90EE90', '#FFD700', '#FFA500',
              '#8B4513', '#4169E1', '#87CEEB', '#808080', '#DC143C',
              '#9370DB', '#FF69B4', '#00CED1', '#FF6347', '#4682B4']
    
    if n_classes > len(colors):
        import matplotlib.cm as cm
        cmap_colors = cm.get_cmap('tab20', n_classes)
        colors = [cmap_colors(i) for i in range(n_classes)]
    
    cmap = ListedColormap(colors[:n_classes])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    
    # 使用中文或英文标签
    if CHINESE_SUPPORT:
        cbar.set_label('土地类别', fontsize=12)
        title = f'分类结果 ({n_classes}类)'
        xlabel = '列'
        ylabel = '行'
        legend_title = '类型'
        class_label = '类别'
    else:
        cbar.set_label('Land Cover Class', fontsize=12)
        title = f'Classification Result ({n_classes} classes)'
        xlabel = 'Column'
        ylabel = 'Row'
        legend_title = 'Type'
        class_label = 'Class'
    
    cbar.set_ticks(unique_classes)
    cbar.set_ticklabels([f'{class_label} {int(i)}' for i in unique_classes])
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 图例
    legend_elements = [
        Patch(facecolor=colors[i], label=f'{class_label} {int(unique_classes[i])}')
        for i in range(n_classes)
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1), title=legend_title, fontsize=10)
    
    plt.tight_layout()
    
    return fig, n_classes

def plot_area_charts_plotly(area_stats):
    """使用Plotly绘制图表（支持中文）"""
    if not area_stats:
        return None, None
    
    df = pd.DataFrame(area_stats)
    
    class_col = get_label('class')
    area_km2_col = get_label('area_km2')
    area_ha_col = get_label('area_ha')
    pct_col = get_label('percentage')
    
    # 饼图
    if CHINESE_SUPPORT:
        pie_title = '各类别面积占比'
    else:
        pie_title = 'Area Distribution by Class'
    
    fig_pie = px.pie(
        df, 
        values=area_km2_col, 
        names=class_col,
        title=pie_title,
        hover_data=[area_ha_col, pct_col]
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    # 柱状图
    if CHINESE_SUPPORT:
        bar_title = '各类别面积分布'
        bar_xlabel = '类别'
        bar_ylabel = '面积 (km²)'
    else:
        bar_title = 'Area Distribution'
        bar_xlabel = 'Class'
        bar_ylabel = 'Area (km²)'
    
    fig_bar = px.bar(
        df,
        x=class_col,
        y=area_km2_col,
        title=bar_title,
        labels={area_km2_col: bar_ylabel, class_col: bar_xlabel},
        text=pct_col
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(showlegend=False)
    
    return fig_pie, fig_bar

# ==================== Streamlit 主界面 ====================

def main():
    # 标题
    st.title(get_label('title'))
    st.markdown(f"### {get_label('subtitle')}")
    
    # 显示字体状态
    if CHINESE_SUPPORT:
        st.sidebar.success("✓ 中文显示已启用")
    else:
        st.sidebar.info("ℹ️ Using English (Chinese font not available)")
    
    # 侧边栏 - 参数设置
    st.sidebar.header(get_label('parameters'))
    
    # 文件上传
    uploaded_file = st.sidebar.file_uploader(
        get_label('upload'), 
        type=['tif', 'tiff'],
        help="Support multi-band GeoTIFF format"
    )
    
    if uploaded_file is not None:
        st.sidebar.success(f"✓ Uploaded: {uploaded_file.name}")
        st.sidebar.info(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
    
    # 分类方法选择
    methods = list_classification_methods()
    method_options = {info['name']: key for key, info in methods.items()}
    
    selected_method_name = st.sidebar.selectbox(
        get_label('method'),
        options=list(method_options.keys()),
        help="Different methods for different scenarios"
    )
    selected_method = method_options[selected_method_name]
    
    st.sidebar.markdown(f"**Description:** {methods[selected_method]['desc']}")
    
    # 类别数设置
    if methods[selected_method]['need_n_clusters']:
        n_clusters = st.sidebar.slider(
            get_label('clusters'),
            min_value=2,
            max_value=15,
            value=6,
            help="Number of land cover classes"
        )
    else:
        n_clusters = 6
        st.sidebar.info("Auto-determine number of classes")
    
    # 后处理选项
    st.sidebar.subheader("Post-processing Options")
    post_process = st.sidebar.checkbox(
        get_label('postprocess'), 
        value=True, 
        help="Remove small patches and smooth results"
    )
    
    min_patch_size = 0
    if post_process:
        min_patch_size = st.sidebar.slider(
            get_label('minpatch'),
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Patches smaller than this will be merged"
        )
    
    # 运行按钮
    st.sidebar.markdown("---")
    run_button = st.sidebar.button(
        get_label('run'), 
        type="primary", 
        use_container_width=True
    )
    
    # 主界面
    if uploaded_file is None:
        # 欢迎界面
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            welcome_text = """
            <div class="info-box">
                <h3>👋 Welcome</h3>
                <p>Professional remote sensing image classification system:</p>
                <ul>
                    <li>✅ Multiple ML algorithms</li>
                    <li>✅ Automatic area calculation</li>
                    <li>✅ Interactive visualization</li>
                    <li>✅ Export & download</li>
                </ul>
                <p><strong>Get Started:</strong> Upload Landsat 8 image on the left</p>
            </div>
            """
            st.markdown(welcome_text, unsafe_allow_html=True)
    
    # 运行分类
    if run_button and uploaded_file is not None:
        st.markdown("---")
        st.header("🔄 Processing")
        
        start_time = time.time()
        
        result = run_classification(
            uploaded_file, 
            selected_method, 
            n_clusters, 
            post_process, 
            min_patch_size
        )
        
        classification_result, valid_mask, area_stats, output_path, pixel_area = result
        
        if classification_result is not None:
            elapsed_time = time.time() - start_time
            
            # 保存到会话状态
            st.session_state.classification_done = True
            st.session_state.classification_result = classification_result
            st.session_state.valid_mask = valid_mask
            st.session_state.area_stats = area_stats
            st.session_state.output_path = output_path
            st.session_state.pixel_area = pixel_area
            
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Classification Complete!</h3>
                <p>Time elapsed: {elapsed_time:.2f} seconds</p>
                <p>Scroll down to view results</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 显示结果
    if st.session_state.classification_done:
        st.markdown("---")
        st.header(get_label('results'))
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            get_label('classmap'),
            get_label('statistics'),
            get_label('datatable'),
            get_label('download')
        ])
        
        with tab1:
            st.subheader("Classification Map")
            
            # 生成图像
            fig, n_classes = plot_classification_matplotlib(
                st.session_state.classification_result,
                st.session_state.valid_mask
            )
            
            st.pyplot(fig)
            plt.close(fig)  # 释放内存
            
            # 显示基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Classes", n_classes)
            with col2:
                valid_pct = np.sum(st.session_state.valid_mask) / st.session_state.valid_mask.size * 100
                st.metric("Valid Area", f"{valid_pct:.1f}%")
            with col3:
                st.metric("Image Size", 
                         f"{st.session_state.classification_result.shape[1]}×{st.session_state.classification_result.shape[0]}")
        
        with tab2:
            st.subheader("Statistical Analysis")
            
            if st.session_state.area_stats:
                # 绘制图表
                fig_pie, fig_bar = plot_area_charts_plotly(st.session_state.area_stats)
                
                if fig_pie and fig_bar:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                # 显示汇总统计
                df_stats = pd.DataFrame(st.session_state.area_stats)
                area_km2_col = get_label('area_km2')
                area_ha_col = get_label('area_ha')
                
                total_area_km2 = df_stats[area_km2_col].sum()
                total_area_ha = df_stats[area_ha_col].sum()
                
                st.markdown("### 📊 Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Area", f"{total_area_km2:.2f} km²")
                with col2:
                    st.metric("Total Area", f"{total_area_ha:.2f} ha")
                with col3:
                    st.metric("Pixel Area", f"{st.session_state.pixel_area:.2f} m²")
                with col4:
                    st.metric("Classes", len(df_stats))
        
        with tab3:
            st.subheader("Data Table")
            
            if st.session_state.area_stats:
                df_stats = pd.DataFrame(st.session_state.area_stats)
                area_km2_col = get_label('area_km2')
                df_stats = df_stats.sort_values(area_km2_col, ascending=False)
                
                st.dataframe(df_stats, use_container_width=True)
                
                st.markdown("### 📈 Statistical Summary")
                st.write(df_stats.describe())
        
        with tab4:
            st.subheader("Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 下载分类结果
                if st.session_state.output_path and os.path.exists(st.session_state.output_path):
                    with open(st.session_state.output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Classification (GeoTIFF)",
                            data=f,
                            file_name="classification_result.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )
            
            with col2:
                # 下载统计数据
                if st.session_state.area_stats:
                    df_stats = pd.DataFrame(st.session_state.area_stats)
                    csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 Download Statistics (CSV)",
                        data=csv,
                        file_name="area_statistics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ==================== 页脚 ====================

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🛰️ Landsat 8 Unsupervised Classification System v3.1</p>
        <p>Built with Streamlit + Scikit-learn + Rasterio</p>
        <p>Algorithms: K-Means | GMM | ISODATA | Birch | Fuzzy C-Means</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()