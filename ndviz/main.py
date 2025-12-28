import math
import numpy as np
import plotly.graph_objects as go

from ndviz import (
    hypercube_pm1,
    hypersphere,
    default_spin,
    random_orthonormal_basis,
    project,
)

def hypercube_edges(n: int):
    edges = []
    for v in range(1 << n):
        for bit in range(n):
            u = v ^ (1 << bit)
            if v < u:
                edges.append((v, u))
    return edges

def build_animation(shape: str, n: int, num_frames: int = 60, num_points: int = 500):
    # формируем исходный набор точек
    if shape == "cube":
        X = hypercube_pm1(n)
    elif shape == "sphere":
        X = hypersphere(n, num_points, seed=0)
    else:
        raise ValueError("shape must be 'cube' or 'sphere'")
    
    X -= np.mean(X, axis=0, keepdims=True)
    spin = default_spin(n, seed=0) if n >= 2 else None
    # Определяем размерность проекции. Если n < 3, используем out_dim = n и
    # ортонормальный базис в виде единичной матрицы; затем дополняем координаты
    # нулями до 3 измерений. Иначе создаём случайный ортонормальный базис в 3D.
    if n >= 3:
        out_dim = 3
        P = random_orthonormal_basis(n, out_dim=out_dim, seed=0)
    else:
        out_dim = n
        # базис идентичности: строки P ортонормальны
        P = np.eye(n)
    frames = []
    ts = np.linspace(0, 2 * math.pi, num_frames)

    for idx, t in enumerate(ts):
        X_rot = spin.apply(X, t) if spin else X
        Y = project(X_rot, P)  # (m,out_dim)
        # при меньшем числе измерений дополняем нулями до 3 для отображения в 3D
        if out_dim < 3:
            pad_width = 3 - out_dim
            Y = np.hstack([Y, np.zeros((Y.shape[0], pad_width))])
        x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]
        traces = []
        traces.append(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            alphahull=0,
            color="rgb(184,134,11)",
            opacity=0.5,
            name="surface",
            showscale=False,
        ))
        if shape == "cube":
            line_x, line_y, line_z = [], [], []
            for a, b in hypercube_edges(n):
                line_x += [x[a], x[b], None]
                line_y += [y[a], y[b], None]
                line_z += [z[a], z[b], None]
            traces.append(go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line=dict(color="rgb(184,134,11)", width=3),
                name="edges",
                hoverinfo="none",
            ))
        frames.append(go.Frame(data=traces, name=str(idx)))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title_text=f"{'Гиперкуб' if shape == 'cube' else 'Гиперсфера'} (n={n})",
        showlegend=False,
        paper_bgcolor="rgb(10,10,10)",
        plot_bgcolor="rgb(10,10,10)",
        scene=dict(
            bgcolor="rgb(10,10,10)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 17, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "bottom",
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": str(i),
                    "method": "animate",
                }
                for i, frame in enumerate(frames)
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "xanchor": "left",
            "y": -0.05,
            "len": 0.9,
        }],
    )
    return fig

def main():
    print("Выберите фигуру:")
    print("  1) Гиперкуб")
    print("  2) Гиперсфера")
    shape = "cube" if int(input("Введите номер: ").strip()) == 1 else "sphere"
    n = int(input("Введите размерность n (>=1): ").strip())
    num_points = min(2000, 500 * n) if shape == "sphere" else 0
    fig = build_animation(shape, n, num_frames=60, num_points=num_points)
    fig.show()

if __name__ == "__main__":
    main()
