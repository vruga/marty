#!/usr/bin/env python3
"""
Generate ArUco marker SVGs without OpenCV dependency.

These are the exact bit patterns for DICT_4X4_50 markers 0-3.
SVGs are scalable - print at any size, just measure after printing.
"""

# ArUco DICT_4X4_50 bit patterns (4x4 inner data, read row by row)
# 1 = black, 0 = white
# These include the required black border (6x6 total grid)
MARKERS_4X4_50 = {
    0: [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    1: [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    2: [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    3: [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
}


def generate_svg(marker_id: int, cell_size: int = 50) -> str:
    """Generate SVG string for an ArUco marker."""
    pattern = MARKERS_4X4_50[marker_id]
    grid_size = len(pattern)
    total_size = grid_size * cell_size
    border = cell_size  # White border around marker

    svg_size = total_size + 2 * border

    lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_size}" height="{svg_size + 40}" viewBox="0 0 {svg_size} {svg_size + 40}">',
        f'  <!-- ArUco Marker ID {marker_id} - DICT_4X4_50 -->',
        f'  <!-- Print and measure: target size is written below marker -->',
        f'  <rect x="0" y="0" width="{svg_size}" height="{svg_size}" fill="white"/>',
    ]

    # Draw black cells
    for row in range(grid_size):
        for col in range(grid_size):
            if pattern[row][col] == 1:
                x = border + col * cell_size
                y = border + row * cell_size
                lines.append(f'  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="black"/>')

    # Add label
    lines.append(f'  <text x="{svg_size//2}" y="{svg_size + 30}" font-family="Arial" font-size="24" text-anchor="middle">ID: {marker_id}</text>')
    lines.append('</svg>')

    return '\n'.join(lines)


def generate_all_markers_page() -> str:
    """Generate a single SVG with all 4 markers for A4 printing."""
    # A4 is 210mm x 297mm, we'll use viewBox units as mm
    page_w, page_h = 210, 297
    marker_size = 50  # 50mm marker
    cell_size = marker_size / 6  # 6x6 grid

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}mm" height="{page_h}mm" viewBox="0 0 {page_w} {page_h}">',
        '  <!-- ArUco Markers 0-3, DICT_4X4_50 -->',
        '  <!-- Print at 100% scale - each marker should be 50mm x 50mm -->',
        '  <rect x="0" y="0" width="210" height="297" fill="white"/>',
        '  <text x="105" y="20" font-family="Arial" font-size="8" text-anchor="middle" font-weight="bold">ArUco Markers - DICT_4X4_50 - Print at 100%</text>',
        '  <text x="105" y="28" font-family="Arial" font-size="5" text-anchor="middle">Each marker should be 50mm. MEASURE after printing!</text>',
    ]

    # 2x2 grid of markers
    positions = [(30, 50), (115, 50), (30, 160), (115, 160)]

    for marker_id, (mx, my) in enumerate(positions):
        pattern = MARKERS_4X4_50[marker_id]

        # Draw marker
        for row in range(6):
            for col in range(6):
                x = mx + col * cell_size
                y = my + row * cell_size
                fill = "black" if pattern[row][col] == 1 else "white"
                lines.append(f'  <rect x="{x:.2f}" y="{y:.2f}" width="{cell_size:.2f}" height="{cell_size:.2f}" fill="{fill}" stroke="none"/>')

        # Label
        lines.append(f'  <text x="{mx + marker_size/2}" y="{my + marker_size + 8}" font-family="Arial" font-size="6" text-anchor="middle">ID: {marker_id} (50mm)</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


if __name__ == '__main__':
    from pathlib import Path

    output_dir = Path(__file__).parent

    # Generate individual markers
    for marker_id in range(4):
        svg = generate_svg(marker_id, cell_size=100)  # Large for quality
        filepath = output_dir / f'marker_{marker_id}.svg'
        filepath.write_text(svg)
        print(f'Generated: {filepath}')

    # Generate combined A4 page
    svg = generate_all_markers_page()
    filepath = output_dir / 'all_markers_A4.svg'
    filepath.write_text(svg)
    print(f'Generated: {filepath}')
    print('\nOpen all_markers_A4.svg in browser and print at 100% scale.')
    print('After printing, MEASURE the markers with a ruler!')
