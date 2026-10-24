#!/usr/bin/env python3
"""Turn measured CSV and raw PGM images into PNG figures."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
with (root / 'measurements.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))

def xy(key):
    good = [(int(r['depth']), float(r[key])) for r in rows if r[key] != 'NA']
    return [v[0] for v in good], [v[1] for v in good]

fig, ax = plt.subplots(figsize=(11, 5.7), dpi=180)
for field, name, marker in [
    ('explicit_ms', 'Explizite Iteration aller $3^n$ Kanten', 'o'),
    ('predicate_ms', 'Mathematischer Full View (512² Pixel)', 's'),
    ('zoom_exact_ms', 'Mathematisches Zoomfenster (512² Pixel)', '^'),
]:
    x, y = xy(field)
    ax.plot(x, y, marker=marker, label=name, linewidth=2.0, markersize=4.0)
ax.set_yscale('log')
ax.set_xlabel('Fraktale Tiefe n')
ax.set_ylabel('Gemessene CPU-Zeit [ms], logarithmische Skala')
ax.set_title('Fraktale Graphen: Kanteniteration vs. direkte Projektion')
ax.set_xlim(4, 26.5)
ax.grid(True, alpha=0.25, which='both')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(root / 'render_times.png', bbox_inches='tight')
plt.close(fig)

files = ['depth9_native.pgm','depth26_full_view.pgm','depth26_zoom_512.pgm']
labels = [
    ('Tiefe 9', '19.683 Kanten, pixelgenau'),
    ('Tiefe 26: Gesamtansicht', '2,54 Billionen Kanten, aggregiert'),
    ('Tiefe 26: Zoom', '512×512-Kachel, pixelgenau'),
]
thumb = 512
pad, gap, head, foot = 30, 24, 90, 43
W = 3*thumb + 2*gap + 2*pad
H = thumb + head + foot
out = Image.new('RGB', (W,H), '#101827')
draw = ImageDraw.Draw(out)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
try:
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 19)
    small = ImageFont.truetype(font_path, 13)
except OSError:
    bold = small = ImageFont.load_default()
for idx, (filename, (title, subt)) in enumerate(zip(files, labels)):
    mask = Image.open(root / filename).convert('L')
    # Distinct image colors are decorative only, occupancy remains binary.
    panel = Image.new('RGB', mask.size, '#162436')
    ink = Image.new('RGB', mask.size, '#86e6d9')
    panel.paste(ink, mask=mask)
    x = pad + idx*(thumb+gap)
    out.paste(panel,(x,head))
    draw.text((x,27), title, font=bold, fill='#f1f6ff')
    draw.text((x,57), subt, font=small, fill='#a5b4ca')
draw.text((pad,H-29), 'Identische lokale Geometrie trotz exponentiell wachsender Kantenanzahl · B⊗n', font=small, fill='#b5c5db')
out.save(root / 'fractal_poc.png', optimize=True)

print('Output files:', root / 'render_times.png', root / 'fractal_poc.png')
