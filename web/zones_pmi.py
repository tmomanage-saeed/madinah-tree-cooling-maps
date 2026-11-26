# Dissolved PMI zones for Map 02, same method as the Map 01 CPI zones:
# 600 m circles around each 1 km cell centre, dissolved per PMI class, higher class wins where they overlap.
import json, arcpy
W = r'C:\GIS\Madinah_Map\web'
SR_W, SR_U = arcpy.SpatialReference(4326), arcpy.SpatialReference(32637)
dp = W + r'\madinah_data.json'
d = json.load(open(dp, encoding='utf-8'))
cells = []
for f in d['hex']['features']:
    r = f['geometry']['coordinates'][0][:-1]
    x = sum(p[0] for p in r) / len(r); y = sum(p[1] for p in r) / len(r)
    pt = arcpy.PointGeometry(arcpy.Point(x, y), SR_W).projectAs(SR_U).firstPoint
    cells.append((pt.X, pt.Y, int(f['properties']['pc'])))
out, higher = [], None
for c in range(5, 1, -1):
    g = None
    for x, y, pc in cells:
        if pc != c: continue
        b = arcpy.PointGeometry(arcpy.Point(x, y), SR_U).buffer(600.0)
        g = b if g is None else g.union(b)
    if g is None: continue
    if higher is not None: g = g.difference(higher)
    higher = g if higher is None else higher.union(g)
    w = g.densify("DISTANCE", 40, 0.5).projectAs(SR_W)
    for part in w:
        rings, cur = [], []
        for p in part:
            if p is None:
                if cur: rings.append(cur); cur = []
                continue
            cur.append([round(p.X, 5), round(p.Y, 5)])
        if cur: rings.append(cur)
        out.append({"type": "Feature", "properties": {"c": c, "pc": c}, "geometry": {"type": "Polygon", "coordinates": rings}})
d['zpmi'] = {"type": "FeatureCollection", "features": out}
json.dump(d, open(dp, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print('zpmi polygons', len(out))
