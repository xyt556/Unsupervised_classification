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
import json


# ==================== 中文字体配置 ====================

# ==================== 中文字体配置 (优化版，适用于云端部署) ====================

def configure_chinese_fonts():
    """
    配置matplotlib中文字体，优先使用项目内置字体文件，确保云端部署时中文正常显示
    """
    import platform
    from matplotlib.font_manager import fontManager, FontProperties
    import os

    # ===== 策略1：优先加载项目内置字体（适用于部署环境） =====
    font_filename = 'SIMSUN.TTC'
    font_path = os.path.join('fonts', font_filename)

    if os.path.exists(font_path):
        try:
            # 动态注册字体到matplotlib
            fontManager.addfont(font_path)

            # 获取字体的实际名称
            prop = FontProperties(fname=font_path)
            font_name = prop.get_name()  # 通常是 'Source Han Sans SC'

            # 设置为matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            return True, font_name
        except Exception as e:
            print(f"⚠️ 加载内置字体失败: {e}")
    else:
        print(f"⚠️ 未找到字体文件: {font_path}")

    # ===== 策略2：回退到系统字体（适用于本地开发） =====
    system = platform.system()
    chinese_fonts = []

    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']

    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available_fonts = {f.name for f in fm.ttflist}

    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # 如果仍然找不到，尝试搜索包含中文关键词的字体
    if selected_font is None:
        for font in available_fonts:
            if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song']):
                selected_font = font
                break

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        return True, selected_font
    else:
        print("❌ 未找到任何可用的中文字体")
        return False, None


# 执行字体配置
CHINESE_SUPPORT, SELECTED_FONT = configure_chinese_fonts()

# ==================== 预定义地物类型模板 ====================

LANDCOVER_TEMPLATES = {
    '水体': {
        'name': '水体',
        'color': '#0066FF',
        'description': '河流、湖泊、水库等水域'
    },
    '植被': {
        'name': '植被',
        'color': '#00AA00',
        'description': '森林、草地、农田等植被覆盖区'
    },
    '城镇': {
        'name': '城镇',
        'color': '#FF0000',
        'description': '建筑、道路、城市建成区'
    },
    '裸地': {
        'name': '裸地',
        'color': '#8B4513',
        'description': '裸土、沙地、未利用地'
    },
    '耕地': {
        'name': '耕地',
        'color': '#FFFF00',
        'description': '农田、耕地、种植用地'
    },
    '森林': {
        'name': '森林',
        'color': '#228B22',
        'description': '乔木林、密林区'
    },
    '草地': {
        'name': '草地',
        'color': '#90EE90',
        'description': '草地、牧场、草原'
    },
    '湿地': {
        'name': '湿地',
        'color': '#00FFFF',
        'description': '沼泽、滩涂、湿地'
    },
    '冰雪': {
        'name': '冰雪',
        'color': '#F0F0F0',
        'description': '积雪、冰川、冰盖'
    },
    '其他': {
        'name': '其他',
        'color': '#808080',
        'description': '其他未分类地物'
    }
}

# 默认颜色方案
DEFAULT_COLORS = [
    '#228B22', '#32CD32', '#90EE90', '#FFD700', '#FFA500',
    '#8B4513', '#4169E1', '#87CEEB', '#808080', '#DC143C',
    '#9370DB', '#FF69B4', '#00CED1', '#FF6347', '#4682B4'
]

# 页面配置
st.set_page_config(
    page_title="遥感影像非监督分类系统",
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
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        margin: 10px 0;
    }
    .class-editor {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #ddd;
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
if 'class_config' not in st.session_state:
    st.session_state.class_config = {}
if 'unique_classes' not in st.session_state:
    st.session_state.unique_classes = []
if 'config_imported' not in st.session_state:
    st.session_state.config_imported = False
if 'editor_initialized' not in st.session_state:
    st.session_state.editor_initialized = False
if 'class_updates' not in st.session_state:
    st.session_state.class_updates = {}

# ==================== 类别配置管理 ====================

def initialize_class_config(unique_classes):
    """初始化类别配置"""
    config = {}
    for i, class_id in enumerate(unique_classes):
        config[int(class_id)] = {
            'name': f'类别 {int(class_id)}',
            'color': DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            'description': ''
        }
    return config

def update_class_config_from_import(imported_config):
    """从导入的配置更新类别配置"""
    if not imported_config:
        return
    
    # 确保配置键是整数类型
    imported_config = {int(k): v for k, v in imported_config.items()}
    
    # 更新现有配置
    for class_id, config in imported_config.items():
        if class_id in st.session_state.class_config:
            st.session_state.class_config[class_id].update(config)
    
    st.session_state.config_imported = True

def apply_template_to_class(class_id, template_key):
    """应用模板到指定类别"""
    if template_key in LANDCOVER_TEMPLATES:
        template_info = LANDCOVER_TEMPLATES[template_key]
        st.session_state.class_config[class_id].update(template_info)
        return True
    return False

def save_class_updates():
    """保存所有类别的更新"""
    for class_id, updates in st.session_state.class_updates.items():
        if class_id in st.session_state.class_config:
            st.session_state.class_config[class_id].update(updates)
    
    # 清空更新缓存
    st.session_state.class_updates = {}

def class_editor_ui():
    """类别编辑器UI - 改进版，解决状态丢失问题"""
    st.markdown("## 🎨 类别自定义设置")
    
    st.info("💡 提示: 您可以为每个类别自定义名称、颜色和描述，也可以从预定义模板中选择地物类型")
    
    # 初始化导入状态标志
    if 'config_imported' not in st.session_state:
        st.session_state.config_imported = False
    
    # 快速设置选项
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚀 快速设置模板")
        st.write("点击下方按钮快速应用预定义的地物类型：")
        
        # 创建模板按钮
        cols = st.columns(5)
        template_keys = list(LANDCOVER_TEMPLATES.keys())
        
        for i, (col, template_key) in enumerate(zip(cols, template_keys[:5])):
            with col:
                template_info = LANDCOVER_TEMPLATES[template_key]
                
                # 显示颜色预览
                st.markdown(f"""
                <div style='text-align: center; margin-bottom: 5px;'>
                    <div style='width: 100%; height: 30px; background-color: {template_info["color"]}; 
                                border-radius: 5px; border: 2px solid #333;'></div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(template_info['name'], key=f"template_top_{i}", use_container_width=True):
                    if i < len(st.session_state.unique_classes):
                        class_id = st.session_state.unique_classes[i]
                        st.session_state.class_config[class_id].update(template_info)
                        st.success(f"已应用模板：{template_info['name']}")
                        st.rerun()
        
        # 第二行模板
        if len(template_keys) > 5:
            cols2 = st.columns(5)
            for i, (col, template_key) in enumerate(zip(cols2, template_keys[5:10]), 5):
                with col:
                    template_info = LANDCOVER_TEMPLATES[template_key]
                    
                    st.markdown(f"""
                    <div style='text-align: center; margin-bottom: 5px;'>
                        <div style='width: 100%; height: 30px; background-color: {template_info["color"]}; 
                                    border-radius: 5px; border: 2px solid #333;'></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(template_info['name'], key=f"template_top_{i}", use_container_width=True):
                        if i < len(st.session_state.unique_classes):
                            class_id = st.session_state.unique_classes[i]
                            st.session_state.class_config[class_id].update(template_info)
                            st.success(f"已应用模板：{template_info['name']}")
                            st.rerun()
    
    with col2:
        st.markdown("### ⚙️ 批量操作")
        
        if st.button("🔄 重置所有设置", use_container_width=True):
            st.session_state.class_config = initialize_class_config(st.session_state.unique_classes)
            st.session_state.config_imported = False  # 重置导入标志
            st.success("已重置所有类别设置")
            st.rerun()
        
        # 导出配置
        config_json = json.dumps(st.session_state.class_config, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 导出配置文件",
            data=config_json,
            file_name="类别配置.json",
            mime="application/json",
            use_container_width=True
        )
        
        # 🔥 修复的导入配置部分
        # 方案1：使用按钮触发导入（推荐）
        st.markdown("#### 📤 导入配置")
        
        # 创建一个唯一的key，每次导入后改变它
        import_key = f"config_upload_{st.session_state.get('import_counter', 0)}"
        
        uploaded_config = st.file_uploader(
            "选择配置文件", 
            type=['json'], 
            key=import_key,
            help="选择之前导出的JSON配置文件"
        )
        
        if uploaded_config is not None:
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("📥 确认导入", type="primary", use_container_width=True):
                    try:
                        # 读取并解析配置
                        config_content = uploaded_config.read()
                        config = json.loads(config_content)
                        
                        # 更新配置
                        st.session_state.class_config = {int(k): v for k, v in config.items()}
                        
                        # 更新导入计数器，这会改变file_uploader的key
                        st.session_state.import_counter = st.session_state.get('import_counter', 0) + 1
                        
                        st.success("✓ 配置导入成功！")
                        time.sleep(0.5)  # 短暂延迟让用户看到成功消息
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 导入失败: {str(e)}")
            
            with col_b:
                if st.button("❌ 取消", use_container_width=True):
                    # 更新计数器以清除文件
                    st.session_state.import_counter = st.session_state.get('import_counter', 0) + 1
                    st.rerun()
    
    st.markdown("---")
    
    # 详细的类别编辑器（其余代码保持不变）
    st.markdown("### 📝 详细编辑")
    
    # 回调函数
    def update_class_name(class_id, key):
        """更新类别名称的回调函数"""
        if key in st.session_state:
            st.session_state.class_config[class_id]['name'] = st.session_state[key]
    
    def update_class_color(class_id, key):
        """更新类别颜色的回调函数"""
        if key in st.session_state:
            st.session_state.class_config[class_id]['color'] = st.session_state[key]
    
    def update_class_desc(class_id, key):
        """更新类别描述的回调函数"""
        if key in st.session_state:
            st.session_state.class_config[class_id]['description'] = st.session_state[key]
    
    # 为每个类别创建编辑器
    for class_id in st.session_state.unique_classes:
        class_id = int(class_id)
        
        with st.expander(f"✏️ {st.session_state.class_config[class_id]['name']} (编号: {class_id})", 
                        expanded=False):
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                # 🔥 关键修改：直接使用 class_config 的值
                name_key = f"name_{class_id}"
                current_name = st.session_state.class_config[class_id]['name']
                
                st.text_input(
                    "类别名称",
                    value=current_name,
                    key=name_key,
                    help="为此类别设置一个有意义的名称",
                    on_change=update_class_name,
                    args=(class_id, name_key)
                )
            
            with col2:
                # 🔥 关键修改：直接使用 class_config 的值
                color_key = f"color_{class_id}"
                current_color = st.session_state.class_config[class_id]['color']
                
                st.color_picker(
                    "显示颜色",
                    value=current_color,
                    key=color_key,
                    help="选择此类别在地图上显示的颜色",
                    on_change=update_class_color,
                    args=(class_id, color_key)
                )
            
            with col3:
                # 从模板选择
                template_choice = st.selectbox(
                    "快速选择模板",
                    options=['手动设置'] + list(LANDCOVER_TEMPLATES.keys()),
                    key=f"template_select_{class_id}",
                    help="从预定义模板中选择地物类型"
                )
                
                if template_choice != '手动设置' and template_choice in LANDCOVER_TEMPLATES:
                    if st.button("应用此模板", key=f"apply_{class_id}", use_container_width=True):
                        template_info = LANDCOVER_TEMPLATES[template_choice]
                        # 只更新 class_config，不要修改组件的 session_state key
                        st.session_state.class_config[class_id].update(template_info)
                        st.rerun()
            
            # 🔥 关键修改：直接使用 class_config 的值
            desc_key = f"desc_{class_id}"
            current_desc = st.session_state.class_config[class_id].get('description', '')
            
            st.text_area(
                "类别描述（可选）",
                value=current_desc,
                key=desc_key,
                height=60,
                help="添加此类别的详细描述",
                on_change=update_class_desc,
                args=(class_id, desc_key)
            )
            
            # 预览
            st.markdown(f"""
            <div style='background-color: {st.session_state.class_config[class_id]["color"]}; 
                        padding: 15px; border-radius: 5px; margin-top: 10px; 
                        border: 2px solid #333; color: white; text-shadow: 1px 1px 2px black;
                        font-size: 16px; font-weight: bold; text-align: center;'>
                预览效果: {st.session_state.class_config[class_id]["name"]}
            </div>
            """, unsafe_allow_html=True)
    
    # 应用更改按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("✅ 应用更改并刷新图表", type="primary", use_container_width=True):
            st.success("✓ 设置已应用！图表将自动更新")
            st.rerun()


# ==================== 示例数据处理 ====================

def load_sample_data():
    """
    从 sample_data 文件夹加载示例遥感影像
    """
    import os

    sample_path = os.path.join('sample_data', 'ylq_L8_2024.tif')

    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"未找到示例数据文件: {sample_path}")

    # 读取文件信息
    with rasterio.open(sample_path) as src:
        info = {
            'width': src.width,
            'height': src.height,
            'bands': src.count,
            'crs': str(src.crs),
            'dtype': src.dtypes[0],
            'bounds': src.bounds
        }

    return sample_path, info


def create_sample_file_object(sample_path):
    """
    将示例文件转换为类似上传文件的对象
    """
    import os

    class SampleFileWrapper:
        def __init__(self, filepath):
            self.name = os.path.basename(filepath)
            self._path = filepath

            # 读取文件内容到内存
            with open(filepath, 'rb') as f:
                self._content = f.read()

            self.size = len(self._content)

        def getvalue(self):
            """返回文件内容（bytes）"""
            return self._content

        def read(self):
            """读取文件内容"""
            return self._content

        def seek(self, pos):
            """模拟文件指针移动"""
            pass

        def tell(self):
            """返回当前文件指针位置"""
            return 0

    return SampleFileWrapper(sample_path)

# ==================== 核心分类函数 ====================

@st.cache_data
def list_classification_methods():
    """列出所有可用的分类方法"""

    # 在函数内部检查依赖是否可用
    try:
        import skfuzzy as fuzz
        fuzzy_available = True
    except ImportError:
        fuzzy_available = False

    methods = {
        'minibatch_kmeans': {
            'name': 'MiniBatch K-Means',
            'desc': '⚡⚡⚡⚡⚡ 最快速，适合大数据集',
            'need_n_clusters': True,
            'available': True
        },
        'kmeans': {
            'name': 'K-Means 聚类',
            'desc': '⚡⚡⚡⚡☆ 速度精度平衡，经典算法',
            'need_n_clusters': True,
            'available': True
        },
        'gmm': {
            'name': '高斯混合模型',
            'desc': '★★★★★ 精度最高，适合精细分类',
            'need_n_clusters': True,
            'available': True
        },
        'birch': {
            'name': 'Birch 聚类',
            'desc': '⚡⚡⚡⚡☆ 内存效率高，适合大数据',
            'need_n_clusters': True,
            'available': True
        },
        'fuzzy_cmeans': {
            'name': '模糊C均值',
            'desc': '★★★★★ 遥感经典算法，软分类',
            'need_n_clusters': True,
            'available': fuzzy_available  # ← 使用局部变量
        },
        'isodata': {
            'name': 'ISODATA',
            'desc': '★★★★☆ 遥感标准算法，可变类别',
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

def calculate_class_areas(classification_result, valid_mask, pixel_area, class_config):
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
        
        class_name = class_config.get(int(cluster_id), {}).get('name', f'类别 {int(cluster_id)}')
        
        area_stats.append({
            '编号': int(cluster_id),
            '类别名称': class_name,
            '像素数量': int(cluster_pixels),
            '面积_km²': round(float(area_km2), 4),
            '面积_公顷': round(float(area_ha), 2),
            '占比_%': round(float(percentage), 2)
        })
    
    return area_stats

def apply_classification(data_clean, n_clusters, method, progress_bar):
    """执行分类"""
    progress_bar.progress(0.3, "正在执行分类算法...")
    
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        progress_bar.progress(0.1, "正在读取影像数据...")
        with rasterio.open(tmp_path) as src:
            bands_data = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
            nodata_value = src.nodata
            
            status_text.success(f"✓ 已加载 {src.count} 个波段，影像尺寸: {src.width}×{src.height}")
        
        progress_bar.progress(0.15, "正在创建数据掩膜...")
        valid_mask_2d = create_valid_data_mask(bands_data, nodata_value)
        
        progress_bar.progress(0.2, "正在进行数据预处理...")
        height, width = bands_data.shape[1], bands_data.shape[2]
        n_bands = bands_data.shape[0]
        
        data_2d = bands_data.reshape(n_bands, -1).T
        valid_mask_1d = valid_mask_2d.ravel()
        data_valid = data_2d[valid_mask_1d]
        
        finite_mask = np.all(np.isfinite(data_valid), axis=1)
        data_clean_temp = data_valid[finite_mask]
        
        if len(data_clean_temp) > 50000:
            status_text.info(f"数据量较大 ({len(data_clean_temp):,} 点)，采样 50000 点进行训练...")
            indices = np.random.choice(len(data_clean_temp), 50000, replace=False)
            data_for_training = data_clean_temp[indices]
        else:
            data_for_training = data_clean_temp
        
        scaler = StandardScaler()
        data_clean = scaler.fit_transform(data_for_training)
        
        # 执行分类
        labels_training, model = apply_classification(data_clean, n_clusters, method, progress_bar)
        
        # 预测全部数据
        if len(data_clean_temp) > 50000:
            progress_bar.progress(0.5, "正在预测全部数据...")
            data_all_scaled = scaler.transform(data_clean_temp)
            if hasattr(model, 'predict'):
                labels_all = model.predict(data_all_scaled)
            else:
                labels_all = labels_training
        else:
            labels_all = labels_training
        
        # 统一将标签从0-based转为1-based
        if labels_all.min() == 0:
            labels_all = labels_all + 1
        
        # 添加调试信息
        unique_labels = np.unique(labels_all)
        status_text.info(f"🔍 分类生成了 {len(unique_labels)} 个类别: {unique_labels}")
        
        progress_bar.progress(0.6, "正在重建分类结果...")
        full_labels_temp = np.zeros(len(data_valid), dtype=np.uint8)
        full_labels_temp[finite_mask] = labels_all
        
        full_labels = np.zeros(height * width, dtype=np.uint8)
        full_labels[valid_mask_1d] = full_labels_temp
        
        classification_result = full_labels.reshape(height, width)
        
        # 再次检查类别数
        unique_before_post = np.unique(classification_result[valid_mask_2d])
        unique_before_post = unique_before_post[unique_before_post > 0]
        status_text.info(f"✓ 后处理前有 {len(unique_before_post)} 个类别")
        
        if post_process and min_patch_size > 0:
            progress_bar.progress(0.7, "正在进行后处理优化...")
            classification_result = simple_post_process(
                classification_result, valid_mask_2d, min_patch_size
            )
            
            # 检查后处理后的类别数
            unique_after_post = np.unique(classification_result[valid_mask_2d])
            unique_after_post = unique_after_post[unique_after_post > 0]
            if len(unique_after_post) < len(unique_before_post):
                status_text.warning(
                    f"⚠️ 后处理移除了 {len(unique_before_post) - len(unique_after_post)} 个类别"
                )
        
        progress_bar.progress(0.8, "正在计算面积统计...")
        pixel_area = calculate_pixel_area(transform, crs)
        
        unique_classes = np.unique(classification_result[valid_mask_2d])
        unique_classes = unique_classes[unique_classes > 0]
        
        # 最终确认类别数
        status_text.success(f"✓ 最终生成 {len(unique_classes)} 个类别: {unique_classes}")
        
        class_config = initialize_class_config(unique_classes)
        
        area_stats = calculate_class_areas(classification_result, valid_mask_2d, pixel_area, class_config)
        
        progress_bar.progress(0.9, "正在保存分类结果...")
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.tif').name
        
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'lzw',
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classification_result, 1)
        
        progress_bar.progress(1.0, "处理完成！")
        
        os.unlink(tmp_path)
        
        return classification_result, valid_mask_2d, area_stats, output_path, pixel_area, unique_classes, class_config
        
    except Exception as e:
        st.error(f"处理失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None, None, None

# ==================== 可视化函数 ====================

def plot_classification_with_custom_colors(classification_result, valid_mask, class_config):
    """生成分类结果图"""
    plot_data = classification_result.copy().astype(float)
    plot_data[~valid_mask] = np.nan
    
    unique_classes = np.unique(classification_result[valid_mask])
    unique_classes = unique_classes[unique_classes > 0]
    n_classes = len(unique_classes)
    
    colors = [class_config.get(int(c), {}).get('color', DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) 
              for i, c in enumerate(unique_classes)]
    
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('土地覆盖类别', fontsize=14, fontweight='bold')
    cbar.set_ticks(unique_classes)
    
    tick_labels = [class_config.get(int(c), {}).get('name', f'类别 {int(c)}') 
                   for c in unique_classes]
    cbar.set_ticklabels(tick_labels)
    
    ax.set_title(f'遥感影像非监督分类结果 ({n_classes}类)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('影像列数（像素）', fontsize=13)
    ax.set_ylabel('影像行数（像素）', fontsize=13)
    
    legend_elements = [
        Patch(facecolor=colors[i], 
              label=class_config.get(int(unique_classes[i]), {}).get('name', f'类别 {int(unique_classes[i])}'),
              edgecolor='black', linewidth=1.5)
        for i in range(n_classes)
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.18, 1), title='地物类型', 
             fontsize=11, title_fontsize=13, framealpha=0.95)
    
    plt.tight_layout()
    
    return fig, n_classes

def plot_area_charts_with_custom_names(area_stats):
    """使用Plotly绘制图表"""
    if not area_stats:
        return None, None
    
    df = pd.DataFrame(area_stats)
    
    colors = [st.session_state.class_config.get(row['编号'], {}).get('color', DEFAULT_COLORS[0]) 
              for _, row in df.iterrows()]
    
    # 饼图
    fig_pie = px.pie(
        df, 
        values='面积_km²', 
        names='类别名称',
        title='各类别面积占比分布',
        hover_data=['面积_公顷', '占比_%'],
        color_discrete_sequence=colors
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12)
    fig_pie.update_layout(font=dict(size=14))
    
    # 柱状图
    fig_bar = go.Figure(data=[
        go.Bar(
            x=df['类别名称'],
            y=df['面积_km²'],
            text=df['占比_%'],
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker=dict(color=colors, line=dict(color='black', width=2))
        )
    ])
    
    fig_bar.update_layout(
        title='各类别面积分布统计',
        xaxis_title='地物类别',
        yaxis_title='面积（平方千米）',
        showlegend=False,
        font=dict(size=14)
    )
    
    return fig_pie, fig_bar

# ==================== Streamlit 主界面 ====================

def main():
    # 标题
    st.title("🛰️ 遥感影像非监督分类系统")
    st.markdown("### 基于机器学习的遥感影像自动分类平台—3S&ML实验室")
    
    # 显示字体状态
    if CHINESE_SUPPORT:
        st.sidebar.success(f"✓ 中文字体已配置: {SELECTED_FONT}")
    else:
        st.sidebar.warning("⚠️ 未检测到中文字体，图表可能显示异常")

    # 侧边栏 - 参数设置
    st.sidebar.header("📋 参数设置")

    # ========== 初始化 session_state ==========
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'sample_file_loaded' not in st.session_state:
        st.session_state.sample_file_loaded = False
    if 'sample_file_info' not in st.session_state:
        st.session_state.sample_file_info = None

    # ========== 数据源选择 ==========
    st.sidebar.markdown("### 📂 数据源")

    data_source = st.sidebar.radio(
        "选择数据来源",
        options=["📁 上传本地文件", "🎯 使用示例数据"],
        label_visibility="collapsed"
    )

    # 切换数据源时重置状态
    if 'last_data_source' not in st.session_state:
        st.session_state.last_data_source = data_source
    elif st.session_state.last_data_source != data_source:
        st.session_state.uploaded_file = None
        st.session_state.sample_file_loaded = False
        st.session_state.sample_file_info = None
        st.session_state.last_data_source = data_source

    # ---------- 选项1：上传本地文件 ----------
    if data_source == "📁 上传本地文件":
        uploaded_file_temp = st.sidebar.file_uploader(
            "选择遥感影像文件",
            type=['tif', 'tiff'],
            help="支持多波段 GeoTIFF 格式的遥感影像",
            key="file_uploader"
        )

        if uploaded_file_temp is not None:
            st.session_state.uploaded_file = uploaded_file_temp
            st.session_state.sample_file_loaded = False

            st.sidebar.success(f"✓ 已上传: {uploaded_file_temp.name}")
            st.sidebar.info(f"📦 文件大小: {uploaded_file_temp.size / (1024 * 1024):.2f} MB")
        elif st.session_state.uploaded_file is None:
            st.sidebar.warning("⚠️ 请上传遥感影像文件")

    # ---------- 选项2：使用示例数据 ----------
    else:
        st.sidebar.info("""
        📊 **示例数据说明**

        - 📐 影像尺寸: 631×508 像素  
        - 📡 波段数量: 7 个多光谱波段  
        - 🗺️ 坐标系统: WGS84  
        - 🌍 包含地物:
          - 💧 水体
          - 🌲 植被  
          - 🏙️ 城镇
          - 🏜️ 裸地
          - 🌾 耕地

        **适合快速体验分类功能！**
        """)

        # 加载示例数据按钮
        if st.sidebar.button(
                "🎯 加载示例数据",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.sample_file_loaded
        ):
            try:
                with st.spinner("正在加载示例遥感影像..."):
                    # 加载示例数据
                    sample_path, sample_info = load_sample_data()
                    sample_file = create_sample_file_object(sample_path)

                    # 保存到 session_state
                    st.session_state.uploaded_file = sample_file
                    st.session_state.sample_file_loaded = True
                    st.session_state.sample_file_info = sample_info

                    st.sidebar.success("✅ 示例数据加载成功！")
                    st.rerun()

            except FileNotFoundError as e:
                st.sidebar.error(f"❌ 错误: {str(e)}")
                st.sidebar.warning("""
                **解决方法:**
                1. 确保 `sample_data/sample_landsat.tif` 文件存在
                2. 检查文件路径是否正确
                3. 重新生成示例数据文件
                """)
            except Exception as e:
                st.sidebar.error(f"❌ 加载失败: {str(e)}")
                import traceback
                with st.sidebar.expander("查看错误详情"):
                    st.code(traceback.format_exc())

        # 显示已加载的示例数据信息
        if st.session_state.sample_file_loaded and st.session_state.sample_file_info:
            info = st.session_state.sample_file_info

            st.sidebar.success("✅ 示例数据已就绪")

            with st.sidebar.expander("📋 数据详细信息", expanded=True):
                st.markdown(f"""
                - **文件名**: {st.session_state.uploaded_file.name}
                - **影像尺寸**: {info['width']} × {info['height']} 像素
                - **波段数量**: {info['bands']} 个
                - **坐标系**: {info['crs']}
                - **数据类型**: {info['dtype']}
                - **文件大小**: {st.session_state.uploaded_file.size / (1024 * 1024):.2f} MB
                """)

    # 获取当前的文件对象
    uploaded_file = st.session_state.uploaded_file

    st.sidebar.markdown("---")
    
    # 分类方法选择
    methods = list_classification_methods()
    method_options = {info['name']: key for key, info in methods.items()}
    
    selected_method_name = st.sidebar.selectbox(
        "选择分类算法",
        options=list(method_options.keys()),
        help="不同算法适用于不同场景，可根据需要选择"
    )
    selected_method = method_options[selected_method_name]
    
    st.sidebar.markdown(f"**算法说明:** {methods[selected_method]['desc']}")
    
    # 类别数设置
    if methods[selected_method]['need_n_clusters']:
        n_clusters = st.sidebar.slider(
            "分类数量",
            min_value=2,
            max_value=15,
            value=6,
            help="将影像划分为几个地物类别"
        )
    else:
        n_clusters = 6
        st.sidebar.info("此算法会自动确定类别数量")
    
    # 后处理选项
    st.sidebar.subheader("🔧 后处理选项")
    post_process = st.sidebar.checkbox(
        "启用后处理优化",
        value=True,
        help="去除小斑块，平滑分类结果，提高分类精度"
    )
    
    min_patch_size = 0
    if post_process:
        min_patch_size = st.sidebar.slider(
            "最小斑块大小（像素）",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="小于此像素数的斑块将被合并到相邻类别"
        )
    
    # 运行按钮
    st.sidebar.markdown("---")
    run_button = st.sidebar.button(
        "🚀 开始分类",
        type="primary",
        use_container_width=True
    )

    # 主界面
    if st.session_state.uploaded_file is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>👋 欢迎使用 遥感影像非监督分类系统</h3>
                <p><strong>系统功能：</strong></p>
                <ul>
                    <li>✅ 支持多种机器学习聚类算法</li>
                    <li>✅ 自动计算各类别面积统计</li>
                    <li>✅ 自定义类别名称和显示颜色</li>
                    <li>✅ 交互式可视化分析</li>
                    <li>✅ 一键导出分类结果</li>
                </ul>
                <p><strong>快速开始：</strong></p>
                <ol>
                    <li>📂 在左侧选择数据来源</li>
                    <li>📁 上传文件 <strong>或</strong> 🎯 加载示例数据</li>
                    <li>⚙️ 设置分类参数</li>
                    <li>🚀 点击"开始分类"按钮</li>
                </ol>
                <p style='color: #666; font-size: 14px; margin-top: 15px;'>
                    💡 <strong>新手提示</strong>：建议先使用<span style='color: #00c853; font-weight: bold;'>示例数据</span>快速体验系统功能
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 使用说明
            with st.expander("📖 使用说明与注意事项"):
                st.markdown("""
                ### 操作步骤
                
                1. **上传影像**
                   - 支持格式：GeoTIFF (.tif, .tiff)
                   - 建议使用Landsat 8地表反射率产品
                   - 文件应包含多个波段数据
                
                2. **选择参数**
                   - 分类算法：根据数据规模和精度要求选择
                   - 分类数量：建议4-8个类别
                   - 后处理：建议启用以提高分类质量
                
                3. **执行分类**
                   - 点击"开始分类"按钮
                   - 等待处理完成（时间取决于影像大小）
                
                4. **自定义类别**
                   - 在类别编辑器中设置名称和颜色
                   - 可选择预定义模板（水体、植被等）
                
                5. **查看结果**
                   - 分类图像：可视化分类结果
                   - 统计分析：查看面积统计图表
                   - 数据表格：详细数据列表
                
                6. **导出结果**
                   - 下载分类结果（GeoTIFF格式）
                   - 下载统计数据（CSV格式）
                   - 导出类别配置（JSON格式）
                
                ### 注意事项
                
                - 📊 数据质量：确保影像数据质量良好，无大面积缺失
                - 🎨 分类数量：不宜过多，一般4-8类效果最佳
                - ⚡ 处理时间：大影像可能需要较长时间，请耐心等待
                - 💾 内存占用：超大影像建议使用MiniBatch K-Means算法
                - 🔍 后处理：建议启用以去除椒盐噪声
                
                ### 算法选择建议
                
                | 场景 | 推荐算法 | 说明 |
                |------|---------|------|
                | 快速预览 | MiniBatch K-Means | 速度最快 |
                | 标准分类 | K-Means | 速度精度平衡 |
                | 高精度 | 高斯混合模型 | 精度最高 |
                | 遥感专业 | ISODATA | 遥感标准算法 |
                """)
    
    # 运行分类
    # 运行分类
    if run_button:
        if st.session_state.uploaded_file is None:
            st.error("❌ 请先上传遥感影像文件或加载示例数据！")
            st.info("👈 请在左侧选择数据来源")
            st.stop()

        st.markdown("---")
        st.header("🔄 正在处理")

        start_time = time.time()

        result = run_classification(
            st.session_state.uploaded_file,  # ← 使用 session_state
            selected_method,
            n_clusters,
            post_process,
            min_patch_size
        )
        
        classification_result, valid_mask, area_stats, output_path, pixel_area, unique_classes, class_config = result
        
        if classification_result is not None:
            elapsed_time = time.time() - start_time
            
            st.session_state.classification_done = True
            st.session_state.classification_result = classification_result
            st.session_state.valid_mask = valid_mask
            st.session_state.area_stats = area_stats
            st.session_state.output_path = output_path
            st.session_state.pixel_area = pixel_area
            st.session_state.unique_classes = unique_classes
            st.session_state.class_config = class_config
            st.session_state.class_updates = {}
            
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ 分类处理完成！</h3>
                <p>⏱️ 总耗时: {elapsed_time:.2f} 秒</p>
                <p>📊 分类类别: {len(unique_classes)} 类</p>
                <p>💡 提示: 向下滚动查看结果，您可以自定义类别名称和颜色</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 显示结果
    if st.session_state.classification_done:
        st.markdown("---")
        
        # 类别编辑器
        with st.expander("🎨 类别自定义编辑器", expanded=True):
            class_editor_ui()
        
        st.markdown("---")
        st.header("📊 分类结果展示")
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ 分类图像",
            "📈 统计分析",
            "📋 数据表格",
            "💾 下载导出"
        ])
        
        with tab1:
            st.subheader("分类结果图像")
            
            fig, n_classes = plot_classification_with_custom_colors(
                st.session_state.classification_result,
                st.session_state.valid_mask,
                st.session_state.class_config
            )
            
            st.pyplot(fig)
            plt.close(fig)
            
            # 基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("分类类别数", n_classes)
            with col2:
                valid_pct = np.sum(st.session_state.valid_mask) / st.session_state.valid_mask.size * 100
                st.metric("有效数据区域", f"{valid_pct:.1f}%")
            with col3:
                st.metric("影像尺寸",
                         f"{st.session_state.classification_result.shape[1]}×{st.session_state.classification_result.shape[0]}")
        
        with tab2:
            st.subheader("面积统计分析")
            
            if st.session_state.area_stats:
                # 更新面积统计（使用最新的类别名称）
                updated_stats = calculate_class_areas(
                    st.session_state.classification_result,
                    st.session_state.valid_mask,
                    st.session_state.pixel_area,
                    st.session_state.class_config
                )
                st.session_state.area_stats = updated_stats
                
                # 绘制图表
                fig_pie, fig_bar = plot_area_charts_with_custom_names(updated_stats)
                
                if fig_pie and fig_bar:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                # 汇总统计
                df_stats = pd.DataFrame(updated_stats)
                total_area_km2 = df_stats['面积_km²'].sum()
                total_area_ha = df_stats['面积_公顷'].sum()
                
                st.markdown("### 📊 汇总统计信息")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总面积", f"{total_area_km2:.2f} km²")
                with col2:
                    st.metric("总面积", f"{total_area_ha:.2f} 公顷")
                with col3:
                    st.metric("单个像素面积", f"{st.session_state.pixel_area:.2f} m²")
                with col4:
                    st.metric("分类类别数", len(df_stats))
        
        with tab3:
            st.subheader("详细数据表格")
            
            if st.session_state.area_stats:
                df_stats = pd.DataFrame(st.session_state.area_stats)
                df_stats = df_stats.sort_values('面积_km²', ascending=False)
                
                st.dataframe(df_stats, use_container_width=True, height=400)
                
                st.markdown("### 📈 数据统计摘要")
                
                # 数值列的统计
                numeric_cols = ['像素数量', '面积_km²', '面积_公顷', '占比_%']
                summary_df = df_stats[numeric_cols].describe()
                st.dataframe(summary_df, use_container_width=True)
        
        with tab4:
            st.subheader("结果文件下载")
            
            st.info("💡 提示：您可以下载分类结果、统计数据和类别配置文件")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📁 分类结果")
                if st.session_state.output_path and os.path.exists(st.session_state.output_path):
                    with open(st.session_state.output_path, 'rb') as f:
                        st.download_button(
                            label="📥 下载分类影像\n(GeoTIFF格式)",
                            data=f,
                            file_name="分类结果.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            help="下载GeoTIFF格式的分类结果，可在GIS软件中打开"
                        )
            
            with col2:
                st.markdown("#### 📊 统计数据")
                if st.session_state.area_stats:
                    df_stats = pd.DataFrame(st.session_state.area_stats)
                    csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 下载面积统计\n(CSV格式)",
                        data=csv,
                        file_name="面积统计.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="下载CSV格式的统计数据，可在Excel中打开"
                    )
            
            with col3:
                st.markdown("#### ⚙️ 配置文件")
                config_json = json.dumps(st.session_state.class_config, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 下载类别配置\n(JSON格式)",
                    data=config_json,
                    file_name="类别配置.json",
                    mime="application/json",
                    use_container_width=True,
                    help="下载类别配置文件，可用于下次分类时导入"
                )
            
            st.markdown("---")
            
            # 生成完整报告
            st.markdown("### 📄 生成分析报告")
            
            if st.button("📝 生成详细分析报告", use_container_width=True):
                report = generate_detailed_report(st.session_state.area_stats, 
                                                 st.session_state.class_config,
                                                 selected_method_name,
                                                 n_clusters)
                
                st.download_button(
                    label="📥 下载分析报告 (TXT格式)",
                    data=report,
                    file_name="分类分析报告.txt",
                    mime="text/plain",
                    use_container_width=True
                )

def generate_detailed_report(area_stats, class_config, method_name, n_clusters):
    """生成详细的分析报告"""
    report = []
    report.append("="*80)
    report.append("遥感影像非监督分类分析报告")
    report.append("="*80)
    report.append(f"\n生成时间: {time.strftime('%Y年%m月%d日 %H:%M:%S')}")
    report.append(f"分类算法: {method_name}")
    report.append(f"目标类别数: {n_clusters}")
    report.append(f"实际生成类别: {len(area_stats)}")
    
    report.append("\n" + "="*80)
    report.append("一、分类结果统计")
    report.append("="*80)
    
    df = pd.DataFrame(area_stats)
    total_area_km2 = df['面积_km²'].sum()
    total_area_ha = df['面积_公顷'].sum()
    
    report.append(f"\n总面积: {total_area_km2:.4f} 平方千米 ({total_area_ha:.2f} 公顷)")
    report.append(f"总像素数: {df['像素数量'].sum():,}")
    
    report.append("\n" + "-"*80)
    report.append("各类别详细信息:")
    report.append("-"*80)
    
    for idx, row in df.iterrows():
        class_id = row['编号']
        class_info = class_config.get(class_id, {})
        
        report.append(f"\n【{row['类别名称']}】 (编号: {class_id})")
        report.append(f"  面积: {row['面积_km²']:.4f} 平方千米")
        report.append(f"       {row['面积_公顷']:.2f} 公顷")
        report.append(f"  像素数量: {row['像素数量']:,}")
        report.append(f"  占比: {row['占比_%']:.2f}%")
        report.append(f"  显示颜色: {class_info.get('color', '未设置')}")
        if class_info.get('description'):
            report.append(f"  描述: {class_info['description']}")
    
    report.append("\n" + "="*80)
    report.append("二、面积统计摘要")
    report.append("="*80)
    
    report.append(f"\n最大类别: {df.loc[df['面积_km²'].idxmax(), '类别名称']} ({df['面积_km²'].max():.4f} km²)")
    report.append(f"最小类别: {df.loc[df['面积_km²'].idxmin(), '类别名称']} ({df['面积_km²'].min():.4f} km²)")
    report.append(f"平均面积: {df['面积_km²'].mean():.4f} km²")
    report.append(f"面积标准差: {df['面积_km²'].std():.4f} km²")
    
    report.append("\n" + "="*80)
    report.append("报告结束")
    report.append("="*80)
    
    return "\n".join(report)

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🛰️ 遥感影像非监督分类系统 v3.2</strong></p>
        <p>基于 Streamlit + Scikit-learn + Rasterio + Matplotlib 开发</p>
        <p>支持的算法: K-Means | GMM | ISODATA | Birch | 模糊C均值</p>
        <p style='font-size: 12px; margin-top: 10px; color: #999;'>
            功能特点: 多算法支持 | 自定义类别 | 面积统计 | 结果导出
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()