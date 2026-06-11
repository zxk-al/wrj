import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ===================== 全局常量配置 =====================
MAX_DRONES = 10000  # 最大支持10000架无人机
DEFAULT_DRONES = 500
SIM_STEP = 0.05      # 移动步长，控制飞行速度

# ===================== 初始化会话状态 =====================
if "drone_pos" not in st.session_state:
    st.session_state.drone_pos = np.zeros((MAX_DRONES, 3))
if "drone_target" not in st.session_state:
    st.session_state.drone_target = np.zeros((MAX_DRONES, 3))
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "drone_num" not in st.session_state:
    st.session_state.drone_num = DEFAULT_DRONES

# ===================== 编队队形生成算法 =====================
def generate_fleet(shape: str, drone_count: int):
    drone_count = min(drone_count, MAX_DRONES)
    pos = np.zeros((MAX_DRONES, 3))
    target = np.zeros((MAX_DRONES, 3))

    if shape == "矩形方阵":
        row = int(np.ceil(np.sqrt(drone_count)))
        idx = 0
        for y in range(row):
            for x in range(row):
                if idx >= drone_count:
                    break
                pos[idx] = [x * 6, y * 6, 0]
                target[idx] = [x * 6, y * 6, 30]
                idx += 1

    elif shape == "环形编队":
        r = drone_count / 4
        for i in range(drone_count):
            angle = 2 * np.pi * i / drone_count
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pos[i] = [x, y, 0]
            target[i] = [x, y, 35]

    elif shape == "一字横队":
        for i in range(drone_count):
            pos[i] = [i * 5, 0, 0]
            target[i] = [i * 5, 0, 28]

    elif shape == "三角编队":
        idx = 0
        layer = 1
        while idx < drone_count:
            offset = (layer - 1) * 3
            for i in range(layer):
                if idx >= drone_count:
                    break
                x = i * 5 - offset
                pos[idx] = [x, layer * 5, 0]
                target[idx] = [x, layer * 5, 25]
                idx += 1
            layer += 1

    elif shape == "随机散点":
        pos[:drone_count] = np.random.rand(drone_count, 3) * 50
        target[:drone_count] = pos[:drone_count] + [0, 0, 22]

    return pos, target

def update_position():
    """更新无人机位置，向目标平滑移动"""
    num = st.session_state.drone_num
    st.session_state.drone_pos[:num] += (st.session_state.drone_target[:num] - st.session_state.drone_pos[:num]) * SIM_STEP

# ===================== 可视化绘制函数 =====================
def draw_2d_view():
    num = st.session_state.drone_num
    x = st.session_state.drone_pos[:num, 0]
    y = st.session_state.drone_pos[:num, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=4, color="#00BFFF"),
        name="无人机集群"
    ))
    fig.update_layout(
        title="无人机编队 2D 俯视图",
        xaxis_title="X 轴", yaxis_title="Y 轴",
        height=400, template="plotly_dark"
    )
    return fig

def draw_3d_view():
    num = st.session_state.drone_num
    x = st.session_state.drone_pos[:num, 0]
    y = st.session_state.drone_pos[:num, 1]
    z = st.session_state.drone_pos[:num, 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=3, color="#39FF14"),
        name="无人机集群"
    ))
    fig.update_layout(
        title="无人机编队 3D 立体视图",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="高度Z"),
        height=500, template="plotly_dark"
    )
    return fig

# ===================== Streamlit 页面主体 =====================
st.set_page_config(page_title="无人机编队仿真系统", layout="wide")
st.title("🚁 云端无人机编队飞行仿真平台")
st.markdown("支持 **1 ~ 10000** 架无人机编队任务，2D/3D 实时可视化")

# 左侧控制面板
with st.sidebar:
    st.header("⚙️ 任务配置")
    drone_count = st.number_input(
        "无人机数量", min_value=1, max_value=MAX_DRONES, value=DEFAULT_DRONES, step=10
    )
    st.session_state.drone_num = drone_count

    shape_list = ["矩形方阵", "环形编队", "一字横队", "三角编队", "随机散点"]
    select_shape = st.selectbox("选择编队队形", shape_list)

    st.divider()
    btn_init = st.button("🔧 初始化编队", use_container_width=True)
    btn_start = st.button("▶️ 开始飞行仿真", use_container_width=True)
    btn_stop = st.button("⏸️ 暂停飞行", use_container_width=True)

# 初始化编队
if btn_init:
    pos, target = generate_fleet(select_shape, drone_count)
    st.session_state.drone_pos = pos
    st.session_state.drone_target = target
    st.session_state.is_running = False
    st.success(f"已完成 {drone_count} 架无人机编队初始化！")

# 控制飞行状态
if btn_start:
    st.session_state.is_running = True
if btn_stop:
    st.session_state.is_running = False

# 分栏展示画面
col1, col2 = st.columns(2)
placeholder_2d = col1.empty()
placeholder_3d = col2.empty()

# 渲染当前画面
placeholder_2d.plotly_chart(draw_2d_view(), use_container_width=True)
placeholder_3d.plotly_chart(draw_3d_view(), use_container_width=True)

# 底部状态信息
st.divider()
st.info(f"当前运行状态：{'飞行中 🟢' if st.session_state.is_running else '已暂停 🔴'} | 在线无人机：{st.session_state.drone_num} 架")

# 安全的循环方式：用rerun代替while循环
if st.session_state.is_running:
    update_position()
    time.sleep(0.1)
    st.rerun()