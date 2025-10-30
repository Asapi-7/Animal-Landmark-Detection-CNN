
import numpy as np

def parse_pts(filename):
    """
    .ptsファイルからランドマークの座標を読み込む関数
    """
    with open(filename) as f:
        rows = [line.strip() for line in f]
    
    # 'n_points'の行を探してポイント数を取得
    n_points_row = [row for row in rows if row.startswith('n_points')]
    if not n_points_row:
        raise ValueError("n_points not found in pts file")
    n_points = int(n_points_row[0].split(':')[1].strip())

    # '{'と'}'の間にある座標を読み込む
    try:
        start = rows.index('{') + 1
        end = rows.index('}')
    except ValueError:
        raise ValueError("Could not find '{' or '}' in pts file")

    coords = []
    for i in range(start, end):
        try:
            x, y = rows[i].split()
            coords.append([float(x), float(y)])
        except ValueError:
            # 空行などをスキップ
            continue
    
    if len(coords) != n_points:
        raise ValueError(f"Expected {n_points} points, but found {len(coords)}")

    return np.array(coords)
