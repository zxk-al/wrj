import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ===================== 全局常量配置 =====================
MAX_DRONES = 10000
DEFAULT_DRONES = 200
SIM_STEP = 0.03  # 控制飞行平滑度
GROUND_Z = 0     # 地面高度
SKY_Z = 80       # 天空上限高度

# ===================== 会话状态初始化 =====================
if "drone_pos" not in st.session_state:
    st.session_state.drone_pos = np.zeros((MAX_DRONES, 3))
if "drone_target" not in st.session_state:
    st.session_state.drone_target = np.zeros((MAX_DRONES, 3))
if "drone_trails" not in st.session_state:
    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "drone_num" not in st.session_state:
    st.session_state.drone_num = DEFAULT_DRONES
if "obstacles" not in st.session_state:
    # 模拟城市建筑/障碍物（中心、半径、高度）
    st.session_state.obstacles = [
        {"pos": (30, 30), "r": 8, "h": 40},
        {"pos": (-20, 10), "r": 6, "h": 35},
        {"pos": (0, -30), "r": 10, "h": 45},
    ]

# ===================== 编队队形生成 =====================
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
                pos[idx] = [x * 6, y * 6, GROUND_Z]
                target[idx] = [x * 6, y * 6, 30]
                idx += 1

    elif shape == "环形编队":
        r = max(10, drone_count / 4)
        for i in range(drone_count):
            angle = 2 * np.pi * i / drone_count
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pos[i] = [x, y, GROUND_Z]
            target[i] = [x, y, 35]

    elif shape == "一字横队":
        for i in range(drone_count):
            pos[i] = [i * 5, 0, GROUND_Z]
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
                pos[idx] = [x, layer * 5, GROUND_Z]
                target[idx] = [x, layer * 5, 25]
                idx += 1
            layer += 1

    elif shape == "随机散点":
        pos[:drone_count] = np.random.rand(drone_count, 3) * 40 - 20
        pos[:drone_count, 2] = GROUND_Z
        target[:drone_count] = pos[:drone_count] + [0, 0, 22]

    # 清空轨迹
    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
    return pos, target

# ===================== 物理更新与避障 =====================
def update_position_with_obstacles():
    num = st.session_state.drone_num
    obstacles = st.session_state.obstacles

    for i in range(num):
        pos = st.session_state.drone_pos[i]
        target = st.session_state.drone_target[i]

        # 基础向目标移动
        dir_vec = target - pos
        dist = np.linalg.norm(dir_vec)
        if dist > 0.1:
            pos += dir_vec * SIM_STEP

        # 障碍物规避（简单排斥力）
        for obs in obstacles:
            ox, oy = obs["pos"]
            obs_r = obs["r"]
            obs_h = obs["h"]
            dx = pos[0] - ox
            dy = pos[1] - oy
            dist_xy = np.sqrt(dx**2 + dy**2)

            if dist_xy < obs_r + 5 and pos[2] < obs_h:
                # 水平推开 + 升高
                push_strength = (obs_r + 5 - dist_xy) / (obs_r + 5) * 0.5
                pos[0] += dx / (dist_xy + 1e-6) * push_strength
                pos[1] += dy / (dist_xy + 1e-6) * push_strength
                pos[2] += 0.2  # 快速升高越过障碍

        # 限制高度范围
        pos[2] = np.clip(pos[2], GROUND_Z, SKY_Z)

        # 记录轨迹（只保留最近100个点）
        st.session_state.drone_trails[i].append(pos.copy())
        if len(st.session_state.drone_trails[i]) > 100:
            st.session_state.drone_trails[i].pop(0)

# ===================== 真实环境3D渲染 =====================
def draw_realistic_scene():
    num = st.session_state.drone_num
    drones = st.session_state.drone_pos[:num]
    trails = st.session_state.drone_trails[:num]
    obstacles = st.session_state.obstacles

    fig = go.Figure()

    # 1. 地面（带网格的城市地面）
    ground_x = np.linspace(-60, 60, 20)
    ground_y = np.linspace(-60, 60, 20)
    gx, gy = np.meshgrid(ground_x, ground_y)
    gz = np.zeros_like(gx)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=gz,
        colorscale=[[0, '#3a3a3a'], [1, '#505050']],
        showscale=False, opacity=0.7, name="地面"
    ))

    # 2. 障碍物（建筑/塔）
    for obs in obstacles:
        ox, oy = obs["pos"]
        r = obs["r"]
        h = obs["h"]
        theta = np.linspace(0, 2*np.pi, 20)
        cx = ox + r * np.cos(theta)
        cy = oy + r * np.sin(theta)
        cz = np.linspace(0, h, 2)
        CX, CZ = np.meshgrid(cx, cz)
        CY, _ = np.meshgrid(cy, cz)
        fig.add_trace(go.Surface(
            x=CX, y=CY, z=CZ,
            colorscale=[[0, '#444444'], [1, '#666666']],
            showscale=False, opacity=0.8, name="建筑"
        ))

    # 3. 无人机轨迹线
    for i in range(num):
        trail = np.array(trails[i])
        if len(trail) > 1:
            fig.add_trace(go.Scatter3d(
                x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
                mode="lines", line=dict(width=1, color="rgba(0,200,255,0.3)"),
                showlegend=False
            ))

    # 4. 无人机本体（带高度颜色渐变）
    fig.add_trace(go.Scatter3d(
        x=drones[:, 0], y=drones[:, 1], z=drones[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=drones[:, 2],
            colorscale="Viridis",
            opacity=0.9,
            line=dict(width=0.5, color="white")
        ),
        name="无人机集群"
    ))

    # 5. 天空背景 + 光照感
    fig.update_layout(
        title="🛸 真实环境无人机编队仿真",
        scene=dict(
            xaxis=dict(range=[-60, 60], title="X (m)"),
            yaxis=dict(range=[-60, 60], title="Y (m)"),
            zaxis=dict(range=[0, SKY_Z], title="高度 Z (m)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=700,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        scene_bgcolor="#16213e"
    )
    return fig

# ===================== 页面主体 =====================
st.set_page_config(page_title="真实环境无人机编队仿真", layout="wide")
st.title("🛸 真实环境无人机编队飞行仿真平台")
st.markdown("支持1~10000架无人机，含城市地面、建筑障碍、轨迹记录与物理避障")

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
    st.success(f"已完成 {drone_count} 架无人机编队初始化，当前环境含{len(st.session_state.obstacles)}个障碍物")

# 控制飞行状态
if btn_start:
    st.session_state.is_running = True
if btn_stop:
    st.session_state.is_running = False

# 主视图区域
placeholder = st.empty()

# 状态信息
st.divider()
col1, col2, col3 = st.columns(3)
col1.info(f"状态：{'飞行中 🟢' if st.session_state.is_running else '已暂停 🔴'}")
col2.info(f"无人机数量：{st.session_state.drone_num} 架")
col3.info(f"障碍物数量：{len(st.session_state.obstacles)} 个")

# 仿真循环（安全rerun方式）
if st.session_state.is_running:
    update_position_with_obstacles()
    placeholder.plotly_chart(draw_realistic_scene(), use_container_width=True)
    time.sleep(0.1)
    st.rerun()
else:
    # 静止状态渲染当前场景
    placeholder.plotly_chart(draw_realistic_scene(), use_container_width=True)
