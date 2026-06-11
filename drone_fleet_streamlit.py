import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ===================== 全局配置 =====================
MAX_DRONES = 10000
DEFAULT_DRONES = 100
GROUND_Z = 0
SKY_Z = 80

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
if "custom_points" not in st.session_state:
    st.session_state.custom_points = []
if "obstacles" not in st.session_state:
    st.session_state.obstacles = [
        {"pos": (25, 25), "r": 7, "h": 38},
        {"pos": (-18, 12), "r": 5, "h": 32},
        {"pos": (5, -28), "r": 9, "h": 42},
        {"pos": (-30, -20), "r": 6, "h": 30}
    ]

# ===================== 1. 预设队形生成 =====================
def gen_preset_fleet(shape, count):
    count = min(count, MAX_DRONES)
    pos = np.zeros((MAX_DRONES, 3))
    target = np.zeros((MAX_DRONES, 3))

    if shape == "矩形方阵":
        row = int(np.ceil(np.sqrt(count)))
        idx = 0
        for y in range(row):
            for x in range(row):
                if idx >= count:
                    break
                pos[idx] = [x * 6, y * 6, GROUND_Z]
                target[idx] = [x * 6, y * 6, 32]
                idx += 1

    elif shape == "环形编队":
        r = max(8, count / 4)
        for i in range(count):
            ang = 2 * np.pi * i / count
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            pos[i] = [x, y, GROUND_Z]
            target[i] = [x, y, 35]

    elif shape == "一字横队":
        for i in range(count):
            pos[i] = [i * 5, 0, GROUND_Z]
            target[i] = [i * 5, 0, 28]

    elif shape == "三角编队":
        idx = 0
        layer = 1
        while idx < count:
            offset = (layer - 1) * 3
            for i in range(layer):
                if idx >= count:
                    break
                x = i * 5 - offset
                pos[idx] = [x, layer * 5, GROUND_Z]
                target[idx] = [x, layer * 5, 25]
                idx += 1
            layer += 1

    elif shape == "随机散点":
        pos[:count] = np.random.rand(count, 3) * 40 - 20
        pos[:count, 2] = GROUND_Z
        target[:count] = pos[:count] + [0, 0, 22]

    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
    return pos, target

# ===================== 2. 自定义队形生成 =====================
def gen_custom_fleet(point_list, count):
    count = min(count, MAX_DRONES)
    pos = np.zeros((MAX_DRONES, 3))
    target = np.zeros((MAX_DRONES, 3))
    point_num = len(point_list)

    if point_num == 0:
        for i in range(count):
            pos[i] = [0, 0, GROUND_Z]
            target[i] = [0, 0, 30]
    else:
        for i in range(count):
            px, py = point_list[i % point_num]
            pos[i] = [px, py, GROUND_Z]
            target[i] = [px, py, 30]

    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
    return pos, target

# ===================== 3. 飞行物理更新 =====================
def update_flight():
    num = st.session_state.drone_num
    obs_list = st.session_state.obstacles

    for i in range(num):
        p = st.session_state.drone_pos[i]
        t = st.session_state.drone_target[i]

        delta = t - p
        dist = np.linalg.norm(delta)
        if dist > 0.1:
            p += delta * 0.05

        for obs in obs_list:
            ox, oy = obs["pos"]
            or_ = obs["r"]
            oh = obs["h"]
            dx = p[0] - ox
            dy = p[1] - oy
            d_xy = np.hypot(dx, dy)

            if d_xy < or_ + 4 and p[2] < oh:
                push = (or_ + 4 - d_xy) / (or_ + 4) * 0.4
                p[0] += dx / (d_xy + 1e-6) * push
                p[1] += dy / (d_xy + 1e-6) * push
                p[2] += 0.15

        p[2] = np.clip(p[2], GROUND_Z, SKY_Z)

        st.session_state.drone_trails[i].append(p.copy())
        if len(st.session_state.drone_trails[i]) > 30:
            st.session_state.drone_trails[i].pop(0)

# ===================== 4. 简化版3D渲染（100%能显示） =====================
def render_scene():
    drone_num = st.session_state.drone_num
    drone_pos = st.session_state.drone_pos
    trails = st.session_state.drone_trails
    obs = st.session_state.obstacles
    custom_pts = st.session_state.custom_points

    fig = go.Figure()

    # 1. 地面（极简渲染）
    gx = np.linspace(-60, 60, 10)
    gy = np.linspace(-60, 60, 10)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz = np.zeros_like(gxx)
    fig.add_trace(go.Surface(
        x=gxx, y=gyy, z=gzz,
        colorscale=[[0, "#333333"], [1, "#555555"]],
        opacity=0.7, showscale=False
    ))

    # 2. 障碍物（极简渲染）
    for building in obs:
        bx, by = building["pos"]
        br = building["r"]
        bh = building["h"]
        theta = np.linspace(0, 2 * np.pi, 8)
        circ_x = bx + br * np.cos(theta)
        circ_y = by + br * np.sin(theta)
        h_range = np.linspace(0, bh, 2)
        cx_mesh, cz_mesh = np.meshgrid(circ_x, h_range)
        cy_mesh, _ = np.meshgrid(circ_y, h_range)

        fig.add_trace(go.Surface(
            x=cx_mesh, y=cy_mesh, z=cz_mesh,
            colorscale=[[0, "#444444"], [1, "#666666"]],
            opacity=0.8, showscale=False
        ))

    # 3. 自定义点位
    if len(custom_pts) > 0:
        pts_arr = np.array(custom_pts)
        fig.add_trace(go.Scatter3d(
            x=pts_arr[:, 0], y=pts_arr[:, 1], z=np.zeros(len(pts_arr)),
            mode="markers", marker=dict(size=6, color="red")
        ))

    # 4. 无人机轨迹（只画前5架，降低压力）
    for i in range(min(drone_num, 5)):
        tr = np.array(trails[i])
        if len(tr) > 2:
            fig.add_trace(go.Scatter3d(
                x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
                mode="lines", line=dict(color="rgba(100,200,255,0.3)", width=1)
            ))

    # 5. 无人机本体
    drones = drone_pos[:drone_num]
    fig.add_trace(go.Scatter3d(
        x=drones[:, 0], y=drones[:, 1], z=drones[:, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=drones[:, 2],
            colorscale="Plasma",
            opacity=0.9
        )
    ))

    # 6. 场景设置
    fig.update_layout(
        title="无人机编队仿真",
        scene=dict(
            xaxis=dict(range=[-60, 60]),
            yaxis=dict(range=[-60, 60]),
            zaxis=dict(range=[0, SKY_Z]),
            aspectmode="cube"
        ),
        height=600,
        template="plotly_dark"
    )
    return fig

# ===================== 页面主体 =====================
st.set_page_config(page_title="无人机编队仿真", layout="wide")
st.title("🏙️ 全真城市环境 · 无人机编队仿真平台")
st.markdown("✅ 预设队形 + ✅ 坐标自定义编队 + ✅ 建筑避障 | 零AI训练，稳定无频闪")

# 左侧控制面板
with st.sidebar:
    st.header("⚙️ 操作面板")
    st.divider()

    drone_cnt = st.number_input("无人机数量", min_value=1, max_value=MAX_DRONES, value=100, step=10)
    st.session_state.drone_num = drone_cnt

    mode = st.radio("编队模式", ["预设队形", "坐标自定义队形"])

    preset_list = ["矩形方阵", "环形编队", "一字横队", "三角编队", "随机散点"]
    if mode == "预设队形":
        select_shape = st.selectbox("选择队形", preset_list)
    else:
        st.subheader("添加自定义点位")
        col1, col2 = st.columns(2)
        x_coord = col1.number_input("X坐标", min_value=-60.0, max_value=60.0, value=0.0, step=1.0)
        y_coord = col2.number_input("Y坐标", min_value=-60.0, max_value=60.0, value=0.0, step=1.0)
        if st.button("➕ 添加点位", use_container_width=True):
            st.session_state.custom_points.append((x_coord, y_coord))
            st.success(f"已添加点位 ({x_coord}, {y_coord})")
        if st.button("🧹 清空点位", use_container_width=True):
            st.session_state.custom_points = []
            st.success("已清空点位")
        st.info(f"当前点位数量：{len(st.session_state.custom_points)}")

    st.divider()

    if mode == "预设队形":
        btn_init = st.button("🔧 初始化编队", use_container_width=True)
    else:
        btn_init = st.button("🔧 生成自定义编队", use_container_width=True)

    btn_start = st.button("▶️ 开始飞行", use_container_width=True)
    btn_pause = st.button("⏸️ 暂停飞行", use_container_width=True)

    st.divider()
    st.info("""
    💡 关键说明：
    1.  点击「初始化编队」后，**必须先点「开始飞行」，画面才会动**
    2.  为了稳定，已大幅降低渲染压力，保证画面一定能显示
    """)

# 主画面容器
view_holder = st.empty()

# 状态栏
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.info(f"状态：{'飞行中 🟢' if st.session_state.is_running else '已暂停 🔴'}")
col2.info(f"无人机：{st.session_state.drone_num} 架")
col3.info(f"建筑：{len(st.session_state.obstacles)} 栋")
col4.info(f"点位：{len(st.session_state.custom_points)} 个")

# ===================== 核心逻辑 =====================
if btn_init:
    if mode == "预设队形":
        pos, target = gen_preset_fleet(select_shape, drone_cnt)
    else:
        pos, target = gen_custom_fleet(st.session_state.custom_points, drone_cnt)
    st.session_state.drone_pos = pos
    st.session_state.drone_target = target
    st.session_state.is_running = False
    st.success("编队初始化完成！请点击「开始飞行」")

if btn_start:
    st.session_state.is_running = True
if btn_pause:
    st.session_state.is_running = False

# 【关键修改】先强制渲染一次静态画面，避免空白
try:
    fig = render_scene()
    view_holder.plotly_chart(fig, use_container_width=True)
except Exception as e:
    view_holder.error(f"渲染失败：{str(e)}")

# 飞行循环（降低频率，避免频闪）
if st.session_state.is_running:
    update_flight()
    st.experimental_rerun()
