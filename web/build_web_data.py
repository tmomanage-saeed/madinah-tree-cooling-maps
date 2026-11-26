# -*- coding: utf-8 -*-
"""
Madinah – Cooling Priority (Task 5) + Pedestrian Movement (Task 6)
Builds the web data from Madinah_City.gdb + Geofabrik and writes the HTML map.
RUN (ArcGIS Pro python):
  "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" build_web_data.py
Output: Madinah_Cooling_Pedestrian_Maps.html (in this folder)
"""
import arcpy, os, json, datetime
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GDB = os.path.join(ROOT, "Madinah_City.gdb")
GF = os.path.join(ROOT, "geofabrik_unzip")
TEMPLATE = os.path.join(HERE, "template.html")
OUT_HTML = os.path.join(HERE, "Madinah_Cooling_Pedestrian_Maps.html")

SR_UTM = arcpy.SpatialReference(32637)
SR_WGS = arcpy.SpatialReference(4326)
arcpy.env.overwriteOutput = True
HARAM = (39.6111, 24.4672)

# ---- Cooling Priority settings ----
SENS_W = {"Mosque": 3.0, "School": 3.0, "Bus": 2.5, "Mall": 2.5, "Park": 1.5}
NABAWI_W = 10.0
SENS_RADIUS_M = 300          # ~4 min walk
W_SENS, W_PED, W_SHADE = 0.40, 0.35, 0.25
PERIM_FOOTPRINT_M = 20       # 20 m ring outside the footprint (forecourt / sidewalk)
PERIM_POINT_M = {"Mosque": 30, "School": 30, "Mall": 40, "Bus": 15}   # zone around the point when there is no footprint
GREEN_FULL = 0.30            # 30 % green in a hexagon = fully shaded

def log(m): print(m, flush=True)

def gf(n): return os.path.join(GF, f"gis_osm_{n}_free_1.shp")

def clean(s):
    s = (s or "").strip()
    return s

def r5(v): return round(v, 5)

def ring_coords(shp):
    """WGS polygon -> GeoJSON coordinates (outer + inner rings, multipart)."""
    polys = []
    for part in shp:
        rings, cur = [], []
        for pt in part:
            if pt is None:
                if cur: rings.append(cur); cur = []
                continue
            cur.append([r5(pt.X), r5(pt.Y)])
        if cur: rings.append(cur)
        if rings: polys.append(rings)
    return polys

def line_coords(shp):
    out = []
    for part in shp:
        c = [[r5(p.X), r5(p.Y)] for p in part if p]
        if len(c) >= 2: out.append(c)
    return out

BND = os.path.join(GDB, "Madinah_City_Boundary")
city = next(r[0] for r in arcpy.da.SearchCursor(BND, ["SHAPE@"], spatial_reference=SR_UTM))
city_area = city.area / 1e6

# =================== HEXAGONS ===================
# 1 km cells (500 m radius from the centre, 86.6 ha hexagon cells), aggregated from the 250 m (5 ha) PMI grid by make_cells_1km.py (area-weighted mean)
HEX = os.path.join(ROOT, "Madinah_Analysis.gdb", "Pedestrian_Movement_Cell_1km")
oids, cx, cy, area, pmi, pcls, wsc, dsc, bsc = [], [], [], [], [], [], [], [], []
with arcpy.da.SearchCursor(HEX, ["OID@", "SHAPE@XY", "SHAPE@AREA", "PMI", "PMI_Class",
                                 "Walk_Score", "Dest_Score", "Built_Score"]) as sc:
    for o, (x, y), a, p, c, w, d, b in sc:
        oids.append(o); cx.append(x); cy.append(y); area.append(a)
        pmi.append(p or 0); pcls.append(c or 1); wsc.append(w or 0); dsc.append(d or 0); bsc.append(b or 0)
oids = np.array(oids); cx = np.array(cx); cy = np.array(cy); area = np.array(area)
pmi = np.array(pmi); pcls = np.array(pcls)
idx = {o: i for i, o in enumerate(oids)}
tree = cKDTree(np.c_[cx, cy])
log(f"Hexagons: {len(oids)}")

def kernel(px, py, pw, r):
    out = np.zeros(len(cx))
    if not px: return out
    pts = np.c_[px, py]; pw = np.array(pw)
    t2 = cKDTree(pts)
    for i, nb in enumerate(t2.query_ball_point(np.c_[cx, cy], r)):
        if nb:
            d2 = ((pts[nb] - [cx[i], cy[i]]) ** 2).sum(1) / (r * r)
            out[i] = ((1 - d2) ** 2 * pw[nb]).sum()
    return out

def norm(v, pct=99):
    pos = v[v > 0]
    hi = np.percentile(pos, pct) if len(pos) else 1
    return np.clip(v / hi, 0, 1)

# =================== SITES ===================
LAYERS = [("Mosques", "Mosque"), ("Schools", "School"), ("Bus_Stations", "Bus"),
          ("Shopping_Malls", "Mall")]
sites = []   # dicts
for fc, cat in LAYERS:
    with arcpy.da.SearchCursor(os.path.join(GDB, fc), ["SHAPE@", "name", "category"],
                               spatial_reference=SR_UTM) as sc:
        for shp, nm, sub in sc:
            nm = clean(nm)
            w = SENS_W[cat]
            if cat == "Mosque" and ("النبوي" in nm or "nabawi" in nm.lower()):
                w = NABAWI_W
            sites.append(dict(cat=cat, name=nm, sub=sub or "", x=shp.firstPoint.X,
                              y=shp.firstPoint.Y, w=w))
with arcpy.da.SearchCursor(os.path.join(GDB, "Parks"), ["SHAPE@", "name"], spatial_reference=SR_UTM) as sc:
    for shp, nm in sc:
        lp = shp.labelPoint
        sites.append(dict(cat="Park", name=clean(nm), sub="park", x=lp.X, y=lp.Y,
                          w=SENS_W["Park"], area=shp.area, geom=shp))
log(f"Sites: {len(sites)}")

# =================== GREEN / SHADE ===================
mem = "memory"
green_src = arcpy.analysis.Select(gf("landuse_a"), mem + "\\lu_sel",
    "fclass IN ('park','grass','forest','orchard','recreation_ground','meadow','allotments','nature_reserve','scrub')")[0]
clip_bnd = arcpy.management.CopyFeatures(BND, mem + "\\bnd")[0]
green = arcpy.analysis.Clip(green_src, clip_bnd, mem + "\\green")[0]
green = arcpy.management.Project(green, os.path.join(mem, "green_utm"), SR_UTM)[0] \
    if arcpy.Describe(green).spatialReference.factoryCode != 32637 else green
gshare = np.zeros(len(oids))
ix = arcpy.analysis.Intersect([[green, ""], [HEX, ""]], mem + "\\gix", "ONLY_FID")[0]
fid = next(f.name for f in arcpy.ListFields(ix) if f.name.upper().startswith("FID_PEDESTRIAN"))
with arcpy.da.SearchCursor(ix, [fid, "SHAPE@AREA"]) as sc:
    for h, a in sc:
        if h in idx: gshare[idx[h]] += a
gshare = np.clip(gshare / area, 0, 1)
tx, ty = [], []
with arcpy.da.SearchCursor(os.path.join(GDB, "Trees_OSM"), ["SHAPE@XY"]) as sc:
    for (x, y), in sc: tx.append(x); ty.append(y)
tree_k = kernel(tx, ty, [1.0] * len(tx), 60)
green_n = np.clip(gshare / GREEN_FULL + 0.15 * tree_k, 0, 1)
shade_def = 1 - green_n
log(f"Green hexes: {(gshare > 0).sum()}")

# =================== COOLING PRIORITY INDEX ===================
sens = kernel([s["x"] for s in sites] + [None] * 0, [s["y"] for s in sites],
              [s["w"] for s in sites], SENS_RADIUS_M)
hp = arcpy.PointGeometry(arcpy.Point(*HARAM), SR_WGS).projectAs(SR_UTM).firstPoint
sens += kernel([hp.X], [hp.Y], [NABAWI_W], SENS_RADIUS_M * 2)
sens_n = norm(sens)
active = (sens_n > 0.01) | (pmi > 0.05)
cpi = np.where(active, W_SENS * sens_n + W_PED * pmi + W_SHADE * shade_def, 0.0)
pos = cpi[cpi > 0]
brk = np.percentile(pos, [40, 65, 85, 95])
ccls = np.where(cpi <= 0, 0, np.digitize(cpi, brk) + 1)
CLABEL = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High", 5: "Critical"}
log("CPI breaks: " + ", ".join(f"{b:.3f}" for b in brk))
for c in range(1, 6):
    log(f"  CPI {c} {CLABEL[c]:<10} {(ccls == c).sum():>6}")

# write to a separate GDB (Madinah_City.gdb may be locked by ArcGIS Pro)
AGDB = os.path.join(ROOT, "Madinah_Analysis.gdb")
if not arcpy.Exists(AGDB):
    arcpy.management.CreateFileGDB(ROOT, "Madinah_Analysis.gdb")
CHEX = os.path.join(AGDB, "Cooling_Priority_Hex")
arcpy.management.CopyFeatures(HEX, CHEX)
for f, t in [("Sens_Score", "DOUBLE"), ("Shade_Deficit", "DOUBLE"), ("Green_Share", "DOUBLE"),
             ("CPI", "DOUBLE"), ("CPI_Class", "SHORT"), ("CPI_Label", "TEXT")]:
    arcpy.management.AddField(CHEX, f, t)
cidx = {}
with arcpy.da.SearchCursor(CHEX, ["OID@", "SHAPE@XY"]) as sc:
    for o, xy in sc: cidx[o] = int(tree.query(xy)[1])
with arcpy.da.UpdateCursor(CHEX, ["OID@", "Sens_Score", "Shade_Deficit", "Green_Share",
                                  "CPI", "CPI_Class", "CPI_Label"]) as uc:
    for r in uc:
        i = cidx[r[0]]; c = int(ccls[i])
        uc.updateRow([r[0], round(float(sens_n[i]), 3), round(float(shade_def[i]), 3),
                      round(float(gshare[i]), 3), round(float(cpi[i]), 3), c, CLABEL.get(c, "None")])

# site priority
_, hi_site = tree.query(np.c_[[s["x"] for s in sites], [s["y"] for s in sites]])
for s, i in zip(sites, hi_site):
    s["cpi"] = float(cpi[i]); s["ccls"] = int(ccls[i]); s["pmi"] = float(pmi[i])
    s["shade"] = float(shade_def[i])
    s["score"] = round(100 * (0.75 * s["cpi"] / max(cpi.max(), 1e-9) + 0.25 * s["w"] / NABAWI_W), 1)
# Site priority tier = percentile of score (so that sites do not all fall into one class)
_sc = np.array([s["score"] for s in sites])
_tb = np.percentile(_sc, [25, 50, 75, 90])
for s in sites:
    s["ccls"] = int(np.digitize(s["score"], _tb) + 1)

# =================== PERIMETERS ===================
log("Perimeters ...")
fp_parts = []
for shp_n, where, cat in [("pofw_a", "fclass LIKE 'muslim%'", "Mosque"),
                          ("pois_a", "fclass IN ('school','kindergarten','college','university')", "School"),
                          ("pois_a", "fclass IN ('mall','department_store')", "Mall"),
                          ("transport_a", "fclass IN ('bus_station')", "Bus")]:
    s = arcpy.analysis.Select(gf(shp_n), f"{mem}\\fp_{cat}_s", where)[0]
    c = arcpy.analysis.Clip(s, clip_bnd, f"{mem}\\fp_{cat}")[0]
    if int(arcpy.management.GetCount(c)[0]):
        fp_parts.append((c, cat))
perims = []   # (cat, name, geom_utm, kind)
footprints = []
for c, cat in fp_parts:
    with arcpy.da.SearchCursor(c, ["SHAPE@", "name"], spatial_reference=SR_UTM) as sc:
        for shp, nm in sc:
            if shp is None or shp.area < 30: continue
            footprints.append((cat, clean(nm), shp))
fp_tree = cKDTree(np.array([[f[2].centroid.X, f[2].centroid.Y] for f in footprints])) if footprints else None
used_fp = set()
for s in sites:
    if s["cat"] == "Park": continue
    pt = arcpy.PointGeometry(arcpy.Point(s["x"], s["y"]), SR_UTM)
    hit = None
    if fp_tree is not None:
        for j in fp_tree.query_ball_point([s["x"], s["y"]], 400):
            if footprints[j][0] == s["cat"] and footprints[j][2].distanceTo(pt) < 5:
                hit = j; break
    if hit is not None and hit not in used_fp:
        used_fp.add(hit)
        g = footprints[hit][2]
        ring = g.buffer(PERIM_FOOTPRINT_M).difference(g)
        perims.append((s, ring, "footprint", g))
    elif hit is None:
        perims.append((s, pt.buffer(PERIM_POINT_M[s["cat"]]), "point", None))
perim_area = sum(p[1].area for p in perims) / 1e6
log(f"Perimeters: {len(perims)}  ({perim_area:.2f} km2)")
PFC = os.path.join(AGDB, "Cooling_Perimeters")
arcpy.management.CreateFeatureclass(AGDB, "Cooling_Perimeters", "POLYGON", spatial_reference=SR_UTM)
for f, t in [("category", "TEXT"), ("name", "TEXT"), ("perim_type", "TEXT"),
             ("CPI_Class", "SHORT"), ("Priority_Score", "DOUBLE")]:
    arcpy.management.AddField(PFC, f, t)
with arcpy.da.InsertCursor(PFC, ["SHAPE@", "category", "name", "perim_type", "CPI_Class", "Priority_Score"]) as ic:
    for s, ring, kind, _ in perims:
        ic.insertRow([ring, s["cat"], s["name"][:250], kind, s["ccls"], s["score"]])

# =================== CORRIDORS (movement) ===================
log("Corridors ...")
corr = []
thr_st = np.percentile(pmi[pmi > 0], 95)
for fc, kind in [("Pedestrian_Network", "ped"), ("Local_Streets", "street")]:
    with arcpy.da.SearchCursor(os.path.join(GDB, fc), ["SHAPE@", "category"], spatial_reference=SR_UTM) as sc:
        for shp, cat in sc:
            if shp is None or shp.length < 25: continue
            mid = shp.positionAlongLine(0.5, True).firstPoint
            _, i = tree.query([mid.X, mid.Y])
            p = pmi[i]
            keep = (kind == "ped" and pcls[i] >= 3) or (kind == "street" and p >= thr_st and shp.length >= 60 and cat in ("tertiary", "residential", "unclassified"))
            if keep:
                corr.append((kind, float(p), shp, cat))
log(f"Corridors kept: {len(corr)}")

# =================== WALKING NETWORK CATCHMENTS (pedestriancatch-style) ===================
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
WALK_SPEED = 1.1          # m/s (heat + pilgrim crowds: 1.04-1.2)
MAX_MIN = 15
log("Walking network ...")
rd = arcpy.analysis.Select(gf("roads"), mem + "\\wk_sel",
     "fclass NOT IN ('motorway','motorway_link','trunk','trunk_link','track','track_grade1',"
     "'track_grade2','track_grade3','track_grade4','track_grade5','bridleway','cycleway','unknown')")[0]
rd = arcpy.analysis.Clip(rd, clip_bnd, mem + "\\wk_clip")[0]
node_id, nx_, ny_ = {}, [], []
def nid(x, y):
    k = (round(x), round(y))
    if k not in node_id:
        node_id[k] = len(nx_); nx_.append(x); ny_.append(y)
    return node_id[k]
ea, eb, el = [], [], []
ehw = {}
_HW0 = {"primary": 10, "primary_link": 5, "secondary": 8, "secondary_link": 5, "tertiary": 6, "tertiary_link": 4, "residential": 4, "unclassified": 4, "living_street": 3, "service": 3, "trunk": 12}
road_feats = []          # (vertex node ids, wgs coords)
with arcpy.da.SearchCursor(rd, ["SHAPE@", "fclass"], spatial_reference=SR_UTM) as sc:
    for shp, _fc in sc:
        _hw = 0 if _fc in ("footway", "pedestrian", "path", "steps") else _HW0.get(_fc, 3)
        if shp is None: continue
        wgs = shp.projectAs(SR_WGS)
        for part, wpart in zip(shp, wgs):
            ids, coords = [], []
            for p, wp in zip(part, wpart):
                if p is None: continue
                ids.append(nid(p.X, p.Y)); coords.append([r5(wp.X), r5(wp.Y)])
            for a, b in zip(ids[:-1], ids[1:]):
                if a == b: continue
                L = ((nx_[a] - nx_[b]) ** 2 + (ny_[a] - ny_[b]) ** 2) ** 0.5
                ea += [a, b]; eb += [b, a]; el += [L, L]; ehw[(min(a, b), max(a, b))] = _hw
            if len(ids) >= 2: road_feats.append((ids, coords))
N = len(nx_)
G = coo_matrix((el, (ea, eb)), shape=(N, N)).tocsr()
nxy = np.c_[nx_, ny_]; ntree = cKDTree(nxy)
log(f"  nodes {N}, edges {len(el)//2}, features {len(road_feats)}")

def snap(points, maxd=150):
    d, j = ntree.query(points)
    return sorted({int(k) for k, dd in zip(j, d) if dd <= maxd})

haram_src = sorted(set(ntree.query_ball_point([hp.X, hp.Y], 220)))  # Haram plaza gates
SCEN = {
    "haram":  haram_src,
    "bus":    snap([[s["x"], s["y"]] for s in sites if s["cat"] == "Bus"]),
    "mosque": snap([[s["x"], s["y"]] for s in sites if s["cat"] == "Mosque"]),
    "school": snap([[s["x"], s["y"]] for s in sites if s["cat"] == "School"]),
    "mall":   snap([[s["x"], s["y"]] for s in sites if s["cat"] == "Mall"]),
}
lim = MAX_MIN * 60 * WALK_SPEED
TT = {}
pred_haram = None
for k, src in SCEN.items():
    if k == "haram":
        dist, pred, _ = dijkstra(G, indices=src, min_only=True, limit=lim, return_predecessors=True)
        pred_haram = pred
    else:
        dist = dijkstra(G, indices=src, min_only=True, limit=lim)
    TT[k] = dist / WALK_SPEED / 60.0      # minutes
    log(f"  {k:<7} sources {len(src):>4}  nodes<=5min {(TT[k]<=5).sum():>6}  <=10 {(TT[k]<=10).sum():>6}")

catch_feats = []
for ids, coords in road_feats:
    pr = {}
    for k in SCEN:
        t = float(np.min(TT[k][ids]))
        if np.isfinite(t) and t <= MAX_MIN: pr[k[0]] = round(t, 1)
    if pr:
        catch_feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": pr})
log(f"  catchment segments: {len(catch_feats)}")

# minute labels (isochrone style, like PedestrianCatch) + per-node walking minutes for click-explore
_A = np.array(ea); _B = np.array(eb); _mpts = []
for k in SCEN:
    t = TT[k]; ta, tb = t[_A], t[_B]
    ok = np.isfinite(ta) & np.isfinite(tb) & (tb > ta)
    for m_ in range(1, MAX_MIN + 1):
        sel = np.where(ok & (ta < m_) & (tb >= m_))[0]
        if not len(sel): continue
        f = (m_ - ta[sel]) / (tb[sel] - ta[sel])
        X = nxy[_A[sel], 0] + (nxy[_B[sel], 0] - nxy[_A[sel], 0]) * f
        Y = nxy[_A[sel], 1] + (nxy[_B[sel], 1] - nxy[_A[sel], 1]) * f
        cell = 170 if m_ % 5 == 0 else 260
        seen = set()
        for x, y in zip(X, Y):
            key = (int(x // cell), int(y // cell))
            if key in seen: continue
            seen.add(key); _mpts.append((x, y, m_, "hbmsl"["haram bus mosque school mall".split().index(k)]))
_mll = []
for c in range(0, len(_mpts), 20000):
    mp = arcpy.Multipoint(arcpy.Array([arcpy.Point(p[0], p[1]) for p in _mpts[c:c + 20000]]), SR_UTM).projectAs(SR_WGS)
    _mll += [(p.X, p.Y) for p in mp]
cmin = {"o": [39.3, 24.2], "p": [v for (lon, lat), q in zip(_mll, _mpts)
        for v in (int(round((lon - 39.3) * 1e5)), int(round((lat - 24.2) * 1e5)), q[2], "hbmsl".index(q[3]))]}
ntime = {k[0] if k != "mall" else "l": [int(round(v * 10)) if np.isfinite(v) and v <= MAX_MIN else -1 for v in TT[k]] for k in SCEN}
log(f"  minute labels {len(_mpts)}")

# coverage: city area (hex) within a 5/10 min walk
_, hex_node_d = None, None
hd, hn = ntree.query(np.c_[cx, cy])
cover = {}
for k in SCEN:
    t = TT[k][hn]; ok = hd < 150
    cover[k] = {str(m): round(float(((t <= m) & ok).sum() * float(area.mean()) / 1e6), 1) for m in (5, 10, 15)}

# pedestrians walking to the Haram (from the shortest-path tree, not invented)
rng = np.random.default_rng(7)
th = TT["haram"]
cand = np.where(np.isfinite(th) & (th >= 6) & (th <= MAX_MIN))[0]
pick = rng.choice(cand, size=min(700, len(cand)), replace=False) if len(cand) else []
trips = []
for n0 in pick:
    path = [int(n0)]
    while pred_haram[path[-1]] >= 0 and len(path) < 5000:
        path.append(int(pred_haram[path[-1]]))
    if len(path) < 3: continue
    wg = [[r5(p.X), r5(p.Y)] for p in arcpy.Multipoint(arcpy.Array([arcpy.Point(nx_[i], ny_[i]) for i in path]), SR_UTM).projectAs(SR_WGS)]
    t0 = float(th[path[0]])
    ts = [round((t0 - float(th[i])) * 60, 1) for i in path]   # seconds since departure
    trips.append({"p": wg, "t": ts})
log(f"  trips to Haram: {len(trips)}")

# City-wide pedestrians: shortest path from any street to the nearest destination (mosque / school / bus / mall / park)
CITY_TRIPS = 5000
_dst = sorted(set(SCEN["mosque"]) | set(SCEN["school"]) | set(SCEN["bus"]) | set(SCEN["mall"]) |
              set(snap([[s["x"], s["y"]] for s in sites if s["cat"] == "Park"])))
_cd, _cp, _ = dijkstra(G, indices=_dst, min_only=True, limit=lim, return_predecessors=True)
_ct = _cd / WALK_SPEED / 60
_ok = np.where(np.isfinite(_ct) & (_ct >= 2) & (_ct <= MAX_MIN))[0]
_, _nh = tree.query(nxy[_ok])
_w = 0.15 + np.clip(pmi[_nh], 0, None) ** 0.7          # more people on busy streets, but some in every area
_w = _w / _w.sum()
_pick = rng.choice(_ok, size=min(CITY_TRIPS, len(_ok)), replace=False, p=_w)
ctrips = []
for n0 in _pick:
    path = [int(n0)]
    while _cp[path[-1]] >= 0 and len(path) < 3000:
        path.append(int(_cp[path[-1]]))
    if len(path) < 3: continue
    ctrips.append(path)
_allc = sorted({i for pth in ctrips for i in pth})
_cmp = arcpy.Multipoint(arcpy.Array([arcpy.Point(nx_[i], ny_[i]) for i in _allc]), SR_UTM).projectAs(SR_WGS)
_cll = {i: (int(round((p.X - 39.3) * 1e5)), int(round((p.Y - 24.2) * 1e5))) for i, p in zip(_allc, _cmp)}
ctrip_out = []
for pth in ctrips:
    t0 = float(_ct[pth[0]])
    ctrip_out.append({"p": [v for i in pth for v in _cll[i]], "t": [int(round((t0 - float(_ct[i])) * 60)) for i in pth]})
log(f"  city-wide trips: {len(ctrip_out)}  (points {sum(len(c['t']) for c in ctrip_out)})")

# =================== PEDESTRIANCATCH KML CROSS-CHECK ===================
import glob, re, xml.etree.ElementTree as ET
pc = {"found": False}
pc_hull_feats, pc_pts = [], []
kmls = sorted(glob.glob(r"C:\Users\saeed\Downloads\*pedcatch*.kml"))
if kmls:
    K = "{http://www.opengis.net/kml/2.2}"
    root = ET.parse(kmls[-1]).getroot()
    starts, marks = [], []
    def parse_t(s):
        m = re.match(r"\s*(\d+)min(?:(\d+)s)?\s+([\d.]+)m\b", s or "")
        return (int(m.group(1)) + int(m.group(2) or 0) / 60, float(m.group(3))) if m else None
    for fo in root.iter(K + "Folder"):
        fname = (fo.findtext(K + "name") or "").strip()
        for pm in fo.findall(K + "Placemark"):
            desc = pm.findtext(K + "description") or ""
            pt = pm.find(".//" + K + "Point/" + K + "coordinates")
            if fname == "Markers" and pt is not None:
                lon, lat = map(float, pt.text.strip().split(",")[:2])
                if desc.strip() == "Start": starts.append((lon, lat))
                else:
                    tv = parse_t(desc)
                    if tv: marks.append((lon, lat, tv[0], tv[1]))
            if fname == "Convex hulls":
                cs = pm.find(".//" + K + "coordinates")
                if cs is None: continue
                ring = [list(map(float, c.split(",")[:2])) for c in cs.text.split()]
                if ring and ring[0][0] < 30:            # fix swapped lat/lon
                    ring = [[b, a] for a, b in ring]
                m = re.match(r"\s*(\d+)min", desc)
                pc_hull_feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                                      "properties": {"m": int(m.group(1)) if m else None}})
    def utm(lon, lat):
        p = arcpy.PointGeometry(arcpy.Point(lon, lat), SR_WGS).projectAs(SR_UTM).firstPoint
        return [p.X, p.Y]
    if starts and marks:
        s_nodes = snap([utm(*s) for s in starts], 300)
        dpc = dijkstra(G, indices=s_nodes, min_only=True, limit=3000)
        mxy = np.array([utm(m[0], m[1]) for m in marks])
        md, mn = ntree.query(mxy)
        ours = dpc[mn] / 1.04 / 60
        theirs = np.array([m[2] for m in marks])
        ok = (md < 25) & np.isfinite(ours)
        diff = ours[ok] - theirs[ok]
        pc = {"found": True, "file": os.path.basename(kmls[-1]), "starts": len(starts), "points": len(marks),
              "compared": int(ok.sum()), "median_abs_min": round(float(np.median(np.abs(diff))), 2),
              "within1": round(100 * float((np.abs(diff) <= 1).mean()), 1),
              "within2": round(100 * float((np.abs(diff) <= 2).mean()), 1),
              "corr": round(float(np.corrcoef(ours[ok], theirs[ok])[0, 1]), 3)}
        pc_pts = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r5(m[0]), r5(m[1])]},
                   "properties": {"t": round(m[2], 1)}} for m in marks]
        pc["start_xy"] = [[r5(a), r5(b)] for a, b in starts]
        log(f"PedestrianCatch cross-check: {pc}")

# PedestrianCatch generated footpath network (sidewalks + crossings with wait time)
pcnet = {"type": "FeatureCollection", "features": []}
gjs = sorted(glob.glob(r"C:\Users\saeed\Downloads\*pedestriancatch*.geojson"))
pc_cross_n = 0
if gjs:
    with open(gjs[-1], encoding="utf-8") as fh:
        g = json.load(fh)
    for f in g.get("features", []):
        if (f.get("geometry") or {}).get("type") != "LineString": continue
        w = 1 if (f.get("properties") or {}).get("wait", 0) else 0
        pc_cross_n += w
        pcnet["features"].append({"type": "Feature", "properties": {"w": w},
            "geometry": {"type": "LineString", "coordinates": [[r5(a), r5(b)] for a, b in f["geometry"]["coordinates"]]}})
    pc["net_file"] = os.path.basename(gjs[-1]); pc["net_n"] = len(pcnet["features"]); pc["cross_n"] = pc_cross_n
    log(f"PedestrianCatch network: {len(pcnet['features'])} segments, crossings {pc_cross_n}")

# =================== SHADE DEMAND -> UMBRELLAS (100 m2) & TREES (53 m2 mean Madinah crown) ===================
import math
# Tree crown: mean crown area of 40,800 single trees measured in Madinah on the Meta/WRI 1 m canopy height map
# (Tolan et al. 2024; canopy >= 3 m, isolated crowns 4-400 m2): mean 53 m2, median 28 m2, mean diameter 7.2 m.
# Spacing = diameter of a 53 m2 crown (8.2 m) rounded to 8 m, so crowns meet at maturity (continuous shade).
UMB_M2, TREE_M2 = 100.0, 53.0
TREE_SP = 8.0
TARGET = {5: 0.60, 4: 0.45, 3: 0.30, 2: 0.15, 1: 0.0}     # shade coverage target by priority tier
CAP_M2 = {"Mosque": 2000, "School": 1500, "Mall": 3000, "Bus": 300, "Park": 3000}  # per-site max shade demand
WALK_WIDTH = 3.0
log("Shade demand ...")
for k, s in enumerate(sites):
    g = arcpy.PointGeometry(arcpy.Point(s["x"], s["y"]), SR_UTM).projectAs(SR_WGS).firstPoint
    s["lon"], s["lat"], s["id"] = r5(g.X), r5(g.Y), k
plan = []          # rows per site / element
umb_pts, tree_pts = [], []

# ---------------- UMBRELLA SUITABILITY MODEL ----------------
# Every 8 m candidate cell gets a score (0-100):
#   S = 30% Heat (CPI) + 30% Footfall (PMI + dwell) + 20% Sun exposure (building side + existing shade) + 20% Access (sidewalk)
# Constraints (cell removed): inside a building, on a road carriageway, under existing tree canopy (<8 m), Haram plaza (existing umbrellas)
# Placement: only S >= SUIT_MIN, 14 m min spacing (10x10 m canopy + 4 m path), up to the site demand, best score first.
from matplotlib.path import Path as MPath
import collections
log("Umbrella suitability ...")
GRID = 8.0
SPACING_U, SPACING_T = 14.0, TREE_SP
SUIT_MIN, SUIT_TREE, SUIT_SHOW = 55, 45, 35
HARAM_EXIST_M = 420      # Masjid an-Nabawi + its plaza (~400 m from centre) already has 250 large umbrellas
SUN_AZ = 255.0          # summer sun 14-16 h from WSW: west/south side of a building is hottest
DWELL = {"Mosque": 1.0, "School": 0.9, "Bus": 1.0, "Mall": 0.8, "Park": 0.7, "Crossing": 1.0, "Junction": 1.0, "Walkway": 0.5}
JUNC_MIN_CPI, JUNC_MIN_S = 2, 40     # one corner umbrella per junction
WALK_INTERVAL = {5: 100, 4: 150, 3: 220}   # one shade stop per this many metres of walkway (~1.5 / 2.3 / 3.3 min)
HW = {"motorway": 12, "motorway_link": 6, "trunk": 12, "trunk_link": 6, "primary": 10, "primary_link": 5,
      "secondary": 8, "secondary_link": 5, "tertiary": 6, "tertiary_link": 4, "residential": 4,
      "unclassified": 4, "living_street": 3, "service": 3}
FOOT = ("footway", "pedestrian", "path", "steps")

# road vertices (4 m densified) with carriageway half width
rd_all = arcpy.analysis.Clip(gf("roads"), clip_bnd, mem + "\\rd_all")[0]
rv, rh, rwk = [], [], []
jdeg = {}
with arcpy.da.SearchCursor(rd_all, ["SHAPE@", "fclass"], spatial_reference=SR_UTM) as sc:
    for shp, fc in sc:
        if shp is None: continue
        hw = 0.0 if fc in FOOT else float(HW.get(fc, 3.0))
        wk = fc not in ("motorway", "motorway_link", "trunk", "trunk_link")
        for part in shp:
            pts = np.array([(p.X, p.Y) for p in part if p])
            if len(pts) < 2: continue
            if hw > 0 and wk:            # junction degree (vehicle roads pedestrians must cross)
                for q, (x, y) in enumerate(pts):
                    kk = (round(x), round(y)); d0 = jdeg.get(kk, (0, 0.0))
                    jdeg[kk] = (d0[0] + (1 if q in (0, len(pts) - 1) else 2), max(d0[1], hw))
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                n = max(1, int(math.hypot(x1 - x0, y1 - y0) // 4))
                f = np.arange(n) / n
                rv.append(np.c_[x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]); rh += [hw] * n; rwk += [wk] * n
rv = np.vstack(rv); rh = np.array(rh); rwk = np.array(rwk); rtree = cKDTree(rv)
log(f"  road vertices {len(rv)}")
otree = cKDTree(np.c_[tx, ty]) if tx else None

def rings_utm(g):
    out = []
    for part in g:
        cur = []
        for p in part:
            if p is None:
                if cur: out.append(cur); cur = []
            else: cur.append((p.X, p.Y))
        if cur: out.append(cur)
    return out

def grid_in(g, step=GRID):
    if g is None or g.area <= 0: return np.zeros((0, 2))
    g = g.densify("DISTANCE", 2, 0.5)          # turn true-curve buffer arcs into segments (otherwise circles come out empty)
    e = g.extent
    xs = np.arange(e.XMin + step / 2, e.XMax, step); ys = np.arange(e.YMin + step / 2, e.YMax, step)
    if not len(xs) or not len(ys): return np.zeros((0, 2))
    X, Y = np.meshgrid(xs, ys); P = np.c_[X.ravel(), Y.ravel()]
    cnt = np.zeros(len(P), int)
    for r in rings_utm(g):
        if len(r) >= 3: cnt += MPath(r).contains_points(P)
    return P[cnt % 2 == 1]

# ---- candidates per element ----
elems = []          # dict(kind, s, cat, P(n,2), side(n), along(n)|None, ring, seglen)
for s, ring, kind, fpg in perims:
    if s["w"] >= NABAWI_W: elems.append(dict(kind="site", s=s, cat=s["cat"], P=np.zeros((0, 2)), side=np.zeros(0), ring=ring)); continue
    if fpg is not None:
        zone = fpg.buffer(PERIM_FOOTPRINT_M).difference(fpg.buffer(2))
        c = fpg.centroid; c0 = (c.X, c.Y)
    else:
        zone = arcpy.PointGeometry(arcpy.Point(s["x"], s["y"]), SR_UTM).buffer(25 if s["cat"] == "Bus" else PERIM_POINT_M[s["cat"]])
        c0 = None
    P = grid_in(zone, 6.0 if s["cat"] == "Bus" else GRID)
    if c0 is not None and len(P):
        az = np.degrees(np.arctan2(P[:, 0] - c0[0], P[:, 1] - c0[1])) % 360
        side = 0.6 + 0.4 * np.cos(np.radians(az - SUN_AZ))
    else:
        side = np.full(len(P), 0.8)
    elems.append(dict(kind="site", s=s, cat=s["cat"], P=P, side=side, ring=ring))
for s in sites:
    if s["cat"] != "Park": continue
    inner = s["geom"].buffer(-3)
    P = grid_in(inner if inner and inner.area > 0 else s["geom"], 10.0)
    elems.append(dict(kind="park", s=s, cat="Park", P=P, side=np.full(len(P), 0.8)))
walk_km = 0.0
with arcpy.da.SearchCursor(os.path.join(GDB, "Pedestrian_Network"), ["SHAPE@"], spatial_reference=SR_UTM) as sc:
    for shp, in sc:
        if shp is None or shp.length < 10: continue
        m = shp.positionAlongLine(0.5, True).firstPoint
        c = int(ccls[int(tree.query([m.X, m.Y])[1])])
        if c < 3: continue
        walk_km += shp.length / 1000
        n = max(1, int(shp.length // GRID))
        al = (np.arange(n) + 0.5) * shp.length / n
        P = np.array([[q.X, q.Y] for q in (shp.positionAlongLine(float(d)).firstPoint for d in al)])
        elems.append(dict(kind="walk", s={"cat": "Walkway", "name": "", "ccls": c}, cat="Walkway", P=P,
                          side=np.full(n, 1.0), along=al, c=c, L=shp.length))
cross_seen = set()
for f in pcnet["features"]:
    if not f["properties"]["w"]: continue
    for lon, lat in (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][-1]):
        p = arcpy.PointGeometry(arcpy.Point(lon, lat), SR_WGS).projectAs(SR_UTM).firstPoint
        key = (round(p.X / 50), round(p.Y / 50))
        if key in cross_seen: continue
        cross_seen.add(key)
        c = int(ccls[int(tree.query([p.X, p.Y])[1])])
        if c < 5: continue
        elems.append(dict(kind="cross", s={"cat": "Crossing", "name": "", "ccls": c}, cat="Crossing",
                          P=np.array([[p.X, p.Y]]), side=np.full(1, 1.0)))
# road junctions: 3+ vehicle-road arms -> pedestrians wait at the corner to cross
_jk = [(k, hw) for k, (d, hw) in jdeg.items() if d >= 3]
_jxy = np.array([k for k, _ in _jk], float); _jhw = np.array([h for _, h in _jk])
_jseen = set(); junc_n = {c: 0 for c in range(0, 6)}
_ang = np.radians(np.arange(0, 360, 45) + 22.5)
for (x, y), hw in zip(_jxy, _jhw):
    key = (int(x // 35), int(y // 35))           # dual-carriageway nodes -> one junction
    if key in _jseen: continue
    _jseen.add(key)
    c = int(ccls[int(tree.query([x, y])[1])])
    if not ((hw >= 6 and c >= 3) or c >= 5): continue    # major-road junction in a High+ zone, or any junction in a Critical zone
    junc_n[c] += 1
    r = hw + 4.0
    P = np.c_[x + r * np.sin(_ang), y + r * np.cos(_ang)]
    elems.append(dict(kind="junc", s={"cat": "Junction", "name": "", "ccls": c}, cat="Junction",
                      P=P, side=np.full(len(P), 1.0)))
log(f"  junctions by CPI class: {junc_n}")

# ---- score all candidates at once ----
eid = np.concatenate([np.full(len(e["P"]), i) for i, e in enumerate(elems)]).astype(int)
CP = np.vstack([e["P"] for e in elems if len(e["P"])])
SIDE = np.concatenate([e["side"] for e in elems])
CAT_ = np.array([elems[i]["cat"] for i in eid])
log(f"  candidates {len(CP)} in {len(elems)} elements")
_, hi = tree.query(CP)
cpos = cpi[cpi > 0]
H = np.clip(cpi[hi] / np.percentile(cpos, 99), 0, 1)
pmi_n = np.clip(pmi[hi] / np.percentile(pmi[pmi > 0], 99), 0, 1)
F = 0.6 * pmi_n + 0.4 * np.array([DWELL[c] for c in CAT_])
nearT = np.full(len(CP), 99.0)
if otree is not None:
    nearT, _ = otree.query(CP, distance_upper_bound=99)
is_park = CAT_ == "Park"
E = SIDE * np.where(is_park, 1.0, 0.5 + 0.5 * shade_def[hi]) * np.where(nearT < 20, 0.6, 1.0)
dd, jj = rtree.query(CP, k=12, distance_upper_bound=40)
valid = np.isfinite(dd)
jj2 = np.where(valid, jj, 0)
on_road = (valid & (rh[jj2] > 0) & (dd < rh[jj2] + 0.5)).any(1)
edge = np.where(valid & rwk[jj2], dd - rh[jj2], 99).min(1)
A = np.clip(1 - (edge - 4) / 20, 0, 1)
S = 100 * (0.30 * H + 0.30 * F + 0.20 * E + 0.20 * A)
hpx = np.hypot(CP[:, 0] - hp.X, CP[:, 1] - hp.Y)
excl = (on_road & (CAT_ != "Crossing")) | (nearT < 8) | (hpx < HARAM_EXIST_M)
# buildings (full OSM layer) + real canopy polygons
cfc = arcpy.management.CreateFeatureclass(mem, "cand", "POINT", spatial_reference=SR_UTM)[0]
arcpy.management.AddField(cfc, "cid", "LONG")
with arcpy.da.InsertCursor(cfc, ["SHAPE@XY", "cid"]) as ic:
    for k, (x, y) in enumerate(CP): ic.insertRow([(x, y), k])
bld = arcpy.analysis.Clip(gf("buildings_a"), clip_bnd, mem + "\\bld")[0]
canopy = arcpy.analysis.Select(gf("landuse_a"), mem + "\\canopy", "fclass IN ('forest','orchard')")[0]
lyr = arcpy.management.MakeFeatureLayer(cfc, "cand_lyr")[0]
for poly, name in [(bld, "building"), (canopy, "canopy")]:
    arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", poly, selection_type="NEW_SELECTION")
    hit = np.array([r[0] for r in arcpy.da.SearchCursor(lyr, ["cid"])], int)
    if len(hit): excl[hit] = True
    log(f"  excluded by {name}: {len(hit)}")
arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
S = np.where(excl, 0, S)
log(f"  suitable (S>={SUIT_MIN}): {(S >= SUIT_MIN).sum()}  excluded {excl.sum()}  road {on_road.sum()}")
start = np.r_[0, np.cumsum([len(e["P"]) for e in elems])]
_empty = collections.Counter(e["cat"] for i, e in enumerate(elems) if e["kind"] == "site" and start[i + 1] == start[i])
log(f"  site elements with no candidate cell: {dict(_empty)}")

# ---- placement ----
storeU, storeT = {}, {}
def free(x, y, sp, st):
    cx_, cy_ = int(x // sp), int(y // sp)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for qx, qy in st.get((cx_ + a, cy_ + b), ()):
                if (qx - x) ** 2 + (qy - y) ** 2 < sp * sp: return False
    return True
def put(x, y, sp, st): st.setdefault((int(x // sp), int(y // sp)), []).append((x, y))
# all umbrellas in a 50 m grid -> "no other umbrella within GAP metres" check
ALLU = {}
GAP = {"junc": 60.0, "cross": 50.0}        # walkway gap = WALK_INTERVAL (100 / 150 / 220 m)
def near_any(x, y, r):
    c = 50.0; cx_, cy_ = int(x // c), int(y // c); kk = int(math.ceil(r / c))
    for a in range(-kk, kk + 1):
        for b in range(-kk, kk + 1):
            for qx, qy in ALLU.get((cx_ + a, cy_ + b), ()):
                if (qx - x) ** 2 + (qy - y) ** 2 < r * r: return True
    return False
def pick(ids, n, thr, sp, st, gap=0.0):
    out = []
    if n <= 0: return out
    for k in ids[np.argsort(-S[ids])]:
        if S[k] < thr: break
        x, y = CP[k]
        if gap and near_any(x, y, gap): continue
        if free(x, y, sp, st):
            put(x, y, sp, st); out.append(k)
            if st is storeU: ALLU.setdefault((int(x // 50), int(y // 50)), []).append((x, y))
        if len(out) >= n: break
    return out

plan = []; umb_pts, tree_pts = [], []
def rec(k, s, el, dest):
    dest.append((float(CP[k, 0]), float(CP[k, 1]), s, el, k))
walk_dem = {"m2": 0, "umb": 0, "tree": 0, "km": round(walk_km, 1)}
cross_umb = 0; junc_umb = {}; junc_skip = {"covered": 0, "nospace": 0}
# order: highest priority first (top sites claim best spots)
order = sorted(range(len(elems)), key=lambda i: (-{"site": 4, "cross": 3, "junc": 2, "park": 1, "walk": 0}[elems[i]["kind"]],
                                                -(elems[i]["s"].get("score") or 0)))
for i in order:
    e = elems[i]; s = e["s"]; ids = np.arange(start[i], start[i + 1])
    tgt = TARGET.get(s["ccls"], 0)
    if e["kind"] in ("site", "park"):
        nab = s.get("w", 0) >= NABAWI_W
        suit = ids[S[ids] >= SUIT_MIN] if len(ids) else ids
        cell = 36.0 if s["cat"] == "Bus" else (100.0 if e["kind"] == "park" else GRID * GRID)
        frac = 0.5 if e["kind"] == "park" else 1.0
        req = 0 if nab else math.ceil(min(len(suit) * cell * tgt * frac, CAP_M2[s["cat"]]) / UMB_M2)
        U = pick(ids, req, SUIT_MIN, SPACING_U, storeU)
        if s["cat"] == "Bus" and not U:          # at least 1 shelter at every bus stop (best open sidewalk cell)
            U = pick(ids, 1, 1, SPACING_U, storeU)
        suitT = ids[S[ids] >= SUIT_TREE] if len(ids) else ids
        reqT = 0 if nab else math.ceil(min(len(suitT) * cell * tgt * frac, CAP_M2[s["cat"]]) / TREE_M2)
        Tt = pick(ids, reqT, SUIT_TREE, SPACING_T, storeT)
        for k in U: rec(k, s, "park" if e["kind"] == "park" else "perimeter", umb_pts)
        for k in Tt: rec(k, s, "park" if e["kind"] == "park" else "perimeter", tree_pts)
        zone = e["ring"].area if e.get("ring") is not None else s.get("area", 0)
        plan.append(dict(type="site", cat=s["cat"], name=s["name"], tier=s["ccls"], score=s["score"],
                         zone_m2=round(zone), suit_m2=round(len(suit) * cell), demand_m2=len(U) * 100,
                         umbrellas=len(U), trees=len(Tt),
                         note=("Inside Haram plaza zone – umbrellas exist" if (nab or math.hypot(s["x"] - hp.X, s["y"] - hp.Y) < HARAM_EXIST_M) and not U
                               else ("No suitable open space (buildings / roads / shade)" if not len(suit) and tgt > 0 else "")),
                         lon=s.get("lon"), lat=s.get("lat"), id=s.get("id")))
    elif e["kind"] == "walk":
        iv = WALK_INTERVAL[e["c"]]
        U = pick(ids, len(ids), 50, SPACING_U, storeU, gap=iv)   # 1 umbrella per iv metres, counting every nearby umbrella
        Tt = pick(ids, len(ids), SUIT_TREE, SPACING_T, storeT)     # trees every 8 m along the walkway
        for k in U: rec(k, s, "walkway", umb_pts)
        for k in Tt: rec(k, s, "walkway", tree_pts)
        walk_dem["umb"] += len(U); walk_dem["tree"] += len(Tt); walk_dem["m2"] += len(U) * 100
    elif e["kind"] == "junc":
        U = pick(ids, 1, JUNC_MIN_S, SPACING_U, storeU, gap=GAP["junc"])
        for k in U: rec(k, s, "junction", umb_pts); junc_umb[s["ccls"]] = junc_umb.get(s["ccls"], 0) + 1
        if not U: junc_skip["covered" if (S[ids] >= JUNC_MIN_S).any() else "nospace"] += 1
    else:
        U = pick(ids, 1, SUIT_MIN, SPACING_U, storeU, gap=GAP["cross"])
        for k in U: rec(k, s, "crossing", umb_pts); cross_umb += 1
log(f"Umbrellas {len(umb_pts)}  Trees {len(tree_pts)}  (walkway umb {walk_dem['umb']}, crossing {cross_umb})")
_ux = np.array([(p[0], p[1]) for p in umb_pts]); _ue = np.array([p[3] for p in umb_pts])
_nd, _ = cKDTree(_ux).query(_ux, k=2)
for _el in ("perimeter", "park", "junction", "crossing", "walkway"):
    m_ = _ue == _el
    if m_.any(): log(f"  QA nearest-umbrella distance [{_el}]: min {_nd[m_,1].min():.0f} m, median {np.median(_nd[m_,1]):.0f} m, n={m_.sum()}")

# ---------------- LONG-TERM STREET TREE PROGRAMME ----------------
# Both kerbs of every walkable street in the city. Tree spacing follows the cooling priority class:
# spacing = crown diameter (8 m) / target kerb shade cover. Critical 100 %, Very High 80 %, High 67 %, Moderate 50 %, Low 33 %.
# (Target cover per class is a planning assumption; the 8 m crown is measured, see TREE_M2.)
ST_SP_CLS = {5: 8.0, 4: 10.0, 3: 12.0, 2: 16.0, 1: 24.0, 0: 24.0}
ST_SP, ST_OFF, ST_MIN_CPI = TREE_SP, 2.5, 3
log("Street trees ...")
jtree = cKDTree(_jxy) if len(_jxy) else None
cand, ckm, km_cls = [], 0.0, {}
with arcpy.da.SearchCursor(rd_all, ["SHAPE@", "fclass"], spatial_reference=SR_UTM) as sc:
    for shp, fc in sc:
        if shp is None or fc in FOOT or fc in ("motorway", "motorway_link", "trunk", "trunk_link", "service", "unknown") \
                or fc.startswith("track") or fc in ("cycleway", "bridleway"): continue
        hw = float(HW.get(fc, 3.0))
        for part in shp:
            P = np.array([(q.X, q.Y) for q in part if q])
            if len(P) < 2: continue
            seg = np.hypot(*np.diff(P, axis=0).T); cum = np.r_[0, np.cumsum(seg)]
            mid = P[len(P) // 2]
            c_ = int(ccls[int(tree.query(mid)[1])]); sp_ = ST_SP_CLS[c_]
            if cum[-1] < sp_: continue
            ckm += cum[-1] / 1000; km_cls[c_] = km_cls.get(c_, 0) + cum[-1] / 1000
            d = np.arange(sp_ / 2, cum[-1], sp_)
            k = np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1)
            f = (d - cum[k]) / np.maximum(seg[k], 1e-6)
            X = P[k, 0] + (P[k + 1, 0] - P[k, 0]) * f; Y = P[k, 1] + (P[k + 1, 1] - P[k, 1]) * f
            ux = (P[k + 1, 0] - P[k, 0]) / np.maximum(seg[k], 1e-6); uy = (P[k + 1, 1] - P[k, 1]) / np.maximum(seg[k], 1e-6)
            o = hw + ST_OFF
            for sg in (1, -1):
                cand.append(np.c_[X - sg * uy * o, Y + sg * ux * o])
SP_ = np.vstack(cand) if cand else np.zeros((0, 2))
log(f"  street candidates {len(SP_)} on {ckm:.0f} km")
ok = np.ones(len(SP_), bool)
ok &= np.hypot(SP_[:, 0] - hp.X, SP_[:, 1] - hp.Y) > HARAM_EXIST_M
if jtree is not None:
    jd, _ = jtree.query(SP_); ok &= jd > 12                         # keep junction sight triangle clear
dd, jj = rtree.query(SP_, k=12, distance_upper_bound=40)
v_ = np.isfinite(dd); jj_ = np.where(v_, jj, 0)
ok &= ~(v_ & (rh[jj_] > 0) & (dd < rh[jj_] + 1.0)).any(1)          # not on another road's carriageway
if otree is not None:
    od, _ = otree.query(SP_); ok &= od > 8                          # tree already exists
SP_ = SP_[ok]
stfc = arcpy.management.CreateFeatureclass(mem, "stc", "POINT", spatial_reference=SR_UTM)[0]
arcpy.management.AddField(stfc, "cid", "LONG")
with arcpy.da.InsertCursor(stfc, ["SHAPE@XY", "cid"]) as ic:
    for k_, (x, y) in enumerate(SP_): ic.insertRow([(float(x), float(y)), k_])
sl_ = arcpy.management.MakeFeatureLayer(stfc, "stc_lyr")[0]
arcpy.management.SelectLayerByLocation(sl_, "INTERSECT", bld, selection_type="NEW_SELECTION")
hitb = np.array([r[0] for r in arcpy.da.SearchCursor(sl_, ["cid"])], int)
keep = np.ones(len(SP_), bool)
if len(hitb): keep[hitb] = False
SP_ = SP_[keep]
street_pts = []
for x, y in SP_:
    if free(x, y, TREE_SP, storeT): put(x, y, TREE_SP, storeT); street_pts.append((float(x), float(y)))
_, shi = tree.query(np.array(street_pts))
st_by_cls = {c: int((ccls[shi] == c).sum()) for c in range(0, 6)}
log(f"  street trees {len(street_pts)}  by CPI class {st_by_cls}  (excluded building {len(hitb)})")
STF = os.path.join(AGDB, "Proposed_Street_Trees")
arcpy.management.CreateFeatureclass(AGDB, "Proposed_Street_Trees", "POINT", spatial_reference=SR_UTM)
arcpy.management.AddField(STF, "CPI_Class", "SHORT")
with arcpy.da.InsertCursor(STF, ["SHAPE@XY", "CPI_Class"]) as ic:
    for (x, y), c in zip(street_pts, ccls[shi]): ic.insertRow([(x, y), int(c)])

# write GDB layers + CSV
for nm, pts in [("Proposed_Umbrellas_100m2", umb_pts), ("Proposed_Site_Trees", tree_pts)]:
    fc = os.path.join(AGDB, nm)
    arcpy.management.CreateFeatureclass(AGDB, nm, "POINT", spatial_reference=SR_UTM)
    for f_, t_ in [("element", "TEXT"), ("category", "TEXT"), ("site_name", "TEXT"), ("CPI_Class", "SHORT"),
                   ("Suitability", "SHORT"), ("S_Heat", "SHORT"), ("S_Footfall", "SHORT"), ("S_SunExp", "SHORT"), ("S_Access", "SHORT")]:
        arcpy.management.AddField(fc, f_, t_)
    with arcpy.da.InsertCursor(fc, ["SHAPE@XY", "element", "category", "site_name", "CPI_Class",
                                    "Suitability", "S_Heat", "S_Footfall", "S_SunExp", "S_Access"]) as ic:
        for x, y, s, el, k in pts:
            ic.insertRow([(x, y), el, s["cat"], (s["name"] or "")[:250], int(s["ccls"]),
                          int(S[k]), int(100 * H[k]), int(100 * F[k]), int(100 * E[k]), int(100 * A[k])])
SFC = os.path.join(AGDB, "Umbrella_Suitability_Cells")
arcpy.management.CreateFeatureclass(AGDB, "Umbrella_Suitability_Cells", "POINT", spatial_reference=SR_UTM)
for f_ in ["Suitability", "S_Heat", "S_Footfall", "S_SunExp", "S_Access"]: arcpy.management.AddField(SFC, f_, "SHORT")
arcpy.management.AddField(SFC, "Excluded", "SHORT")
with arcpy.da.InsertCursor(SFC, ["SHAPE@XY", "Suitability", "S_Heat", "S_Footfall", "S_SunExp", "S_Access", "Excluded"]) as ic:
    for k in range(len(CP)):
        ic.insertRow([tuple(CP[k]), int(S[k]), int(100 * H[k]), int(100 * F[k]), int(100 * E[k]), int(100 * A[k]), int(excl[k])])
import csv
plan_csv = os.path.join(ROOT, "Action_Requirement_Plan.csv")
with open(plan_csv, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["Rank", "Category", "Name", "Priority tier", "Priority score", "Zone m2", "Suitable open space m2",
                "Shade m2", "Umbrellas (100 m2) short-term", "Shade trees (53 m2 crown) long-term", "Note"])
    for k, r in enumerate(sorted(plan, key=lambda r: -r["score"]), 1):
        w.writerow([k, r["cat"], r["name"], CLABEL[r["tier"]], r["score"], r["zone_m2"], r["suit_m2"], r["demand_m2"],
                    r["umbrellas"], r["trees"], r["note"]])
    w.writerow([])
    w.writerow(["", "Priority walkways (CPI High+)", f"{walk_dem['km']:.1f} km", "", "", "", "", round(walk_dem["m2"]),
                walk_dem["umb"], walk_dem["tree"], "1 shade stop every 100 / 150 / 220 m (Critical / Very High / High)"])
    w.writerow(["", "Road crossings (CPI Critical)", "", "", "", "", "", cross_umb * 100, cross_umb, "",
                "1 umbrella per crossing waiting point (PedestrianCatch network)"])
    w.writerow(["", "Road junctions (major road in High+ zone, all in Critical)", sum(junc_n.values()), "", "", "", "",
                sum(junc_umb.values()) * 100, sum(junc_umb.values()), "",
                "1 corner umbrella per junction (3+ vehicle-road arms) where open sidewalk space exists"])
    w.writerow(["", "Street trees long-term (all zones, both sides)", f"{ckm:.0f} km", "", "", "", "",
                round(len(street_pts) * TREE_M2), "", len(street_pts), f"every 8-24 m by priority, {ST_OFF} m outside kerb; not at junctions / Haram plaza"])

def to_wgs_many(xy):
    out = []
    for c in range(0, len(xy), 20000):
        mp = arcpy.Multipoint(arcpy.Array([arcpy.Point(float(x), float(y)) for x, y in xy[c:c + 20000]]), SR_UTM).projectAs(SR_WGS)
        out += [(p.X, p.Y) for p in mp]
    return out
def feats(pts, full):
    if not pts: return []
    ll = to_wgs_many([(p[0], p[1]) for p in pts])
    out = []
    for (lon, lat), (x, y, s, el, k) in zip(ll, pts):
        pr = {"e": el[0], "cat": s["cat"], "c": int(s["ccls"])}
        if full: pr.update(s=int(S[k]), h=int(100 * H[k]), f=int(100 * F[k]), x=int(100 * E[k]), a=int(100 * A[k]), n=(s["name"] or "")[:60])
        out.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r5(lon), r5(lat)]}, "properties": pr})
    return out
umb_feats = feats(umb_pts, True)
tree_feats = feats(tree_pts, False)
# suitability surface (compact: dx, dy in 1e-5 deg from origin, score)
_sk = np.where(S >= SUIT_SHOW)[0]
_sll = to_wgs_many(CP[_sk])
suit = {"o": [39.3, 24.2], "p": [v for (lon, lat), k in zip(_sll, _sk)
                                  for v in (int(round((lon - 39.3) * 1e5)), int(round((lat - 24.2) * 1e5)), int(S[k]))]}
log(f"  suitability cells exported: {len(_sk)}")
_stll = to_wgs_many(street_pts)
stree = {"o": [39.3, 24.2], "p": [v for lon, lat in _stll for v in (int(round((lon - 39.3) * 1e5)), int(round((lat - 24.2) * 1e5)))]}
summ = {}
for r in plan:
    a = summ.setdefault(r["cat"], {"sites": 0, "crit": 0, "m2": 0, "umb": 0, "tree": 0})
    a["sites"] += 1; a["crit"] += r["tier"] >= 4; a["m2"] += r["demand_m2"]; a["umb"] += r["umbrellas"]; a["tree"] += r["trees"]
summ["Walkway"] = {"sites": None, "km": round(walk_dem["km"], 1), "m2": round(walk_dem["m2"]), "umb": walk_dem["umb"], "tree": walk_dem["tree"]}
summ["Crossing"] = {"sites": sum(1 for e in elems if e["kind"] == "cross"), "m2": cross_umb * 100, "umb": cross_umb, "tree": 0}
_ju = sum(junc_umb.values())
summ["Junction"] = {"sites": sum(junc_n.values()), "m2": _ju * 100, "umb": _ju, "tree": 0, "covered": junc_skip["covered"], "nospace": junc_skip["nospace"]}
log(f"  junctions skipped: {junc_skip}")
summ["StreetTree"] = {"sites": None, "km": round(ckm), "m2": round(len(street_pts) * TREE_M2), "umb": 0, "tree": len(street_pts)}
plan_top = [dict(r, tl=CLABEL[r["tier"]]) for r in sorted(plan, key=lambda r: -r["score"])[:40]]

# =================== ROUTING GRAPH (browser A->B walking time) ===================
gl = [None] * N
_wg = arcpy.Multipoint(arcpy.Array([arcpy.Point(x, y) for x, y in zip(nx_, ny_)]), SR_UTM).projectAs(SR_WGS)
# Multipoint order is preserved
_wl = [(p.X, p.Y) for p in _wg]
LON0, LAT0 = 39.3, 24.2
node_q = []
for x, y in _wl:
    node_q += [int(round((x - LON0) * 1e5)), int(round((y - LAT0) * 1e5))]
_seen = set(); edge_q = []
for a, b in zip(ea, eb):
    if a < b and (a, b) not in _seen:
        _seen.add((a, b)); edge_q += [a, b]
# edge heat factor (cool route): hot priority zones costlier, green cheaper
_ea = np.array(edge_q[0::2]); _eb = np.array(edge_q[1::2])
_mx = (np.array(nx_)[_ea] + np.array(nx_)[_eb]) / 2; _my = (np.array(ny_)[_ea] + np.array(ny_)[_eb]) / 2
_d, _hi = tree.query(np.c_[_mx, _my])
_c = np.where(_d < 200, ccls[_hi], 0)
_f = 1 + np.select([_c >= 5, _c == 4, _c == 3], [0.9, 0.6, 0.25], 0) - np.minimum(0.3, np.where(_d < 200, gshare[_hi], 0))
graph = {"o": [LON0, LAT0], "n": node_q, "e": edge_q, "h": [int(round(v * 100)) for v in _f],
         "w": [ehw.get((a, b), 3) for a, b in zip(edge_q[0::2], edge_q[1::2])]}
# all OSM buildings (route tree/umbrella suggestions must not fall on a building)
bld_rings = []
with arcpy.da.SearchCursor(bld, ["SHAPE@"], spatial_reference=SR_WGS) as sc:
    for shp, in sc:
        if shp is None: continue
        for poly in ring_coords(shp):
            r0 = poly[0]
            bld_rings.append([v for x, y in r0 for v in (int(round((x - LON0) * 1e5)), int(round((y - LAT0) * 1e5)))])
log(f"Buildings exported: {len(bld_rings)} rings")
# WRI / Engel et al. 2026 air-temp assumption: -0.33 C per +10 pp canopy
# canopy added and air cooling per priority class (trees counted in the class where they stand)
_all_t = np.array([(p[0], p[1]) for p in tree_pts] + list(street_pts))
_, _thi = tree.query(_all_t)
_tcls = ccls[_thi]
tiers = {}
for c in range(0, 6):
    a_ = float((ccls == c).sum() * area.mean())
    n_ = int((_tcls == c).sum())
    pp_ = 100 * n_ * TREE_M2 / a_ if a_ else 0
    tiers[c] = {"sp": ST_SP_CLS[c], "km": round(km_cls.get(c, 0)), "n": n_, "km2": round(a_ / 1e6, 1),
                "canopy_pp": round(pp_, 2), "dT": round(-0.33 * pp_ / 10, 2)}
    log(f"  tier {c}: spacing {ST_SP_CLS[c]} m, {n_} trees, +{pp_:.2f} pp canopy, air {-0.33*pp_/10:.2f} C")
hot = _tcls >= ST_MIN_CPI
hot_area_m2 = float(((ccls >= ST_MIN_CPI).sum()) * area.mean())     # High+ zones
tree_canopy_m2 = int(hot.sum()) * TREE_M2
canopy_pp = 100 * tree_canopy_m2 / hot_area_m2
air_dT = -0.33 * canopy_pp / 10
ped_space_m2 = perim_area * 1e6 + walk_dem["km"] * 1000 * WALK_WIDTH + (cross_umb + _ju) * 100
shade_m2 = sum(r["demand_m2"] for r in plan) + walk_dem["m2"] + (cross_umb + _ju) * 100
log(f"Canopy +{canopy_pp:.2f} pp in High+ zones -> air {air_dT:.2f} C ; shaded ped space {100*shade_m2/ped_space_m2:.0f}%")
log(f"Graph nodes {N} edges {len(edge_q)//2}  multipoint {len(_wl)}")

# =================== EXPORT ===================
def to_wgs(g): return g.projectAs(SR_WGS)

hex_feats = []
tree_ha = np.bincount(_thi, minlength=len(oids)) / (area / 1e4)   # proposed trees per hectare
with arcpy.da.SearchCursor(HEX, ["OID@", "SHAPE@"], spatial_reference=SR_WGS) as sc:
    for o, shp in sc:
        i = idx[o]
        if cpi[i] <= 0 and pmi[i] <= 0: continue
        ring = ring_coords(shp)[0][0]
        hex_feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"c": int(ccls[i]), "v": round(float(cpi[i]), 3),
                                         "p": round(float(pmi[i]), 3), "pc": int(pcls[i]),
                                         "s": round(float(sens_n[i]), 2), "sh": round(float(shade_def[i]), 2),
                                         "w": round(float(wsc[i]), 2), "d": round(float(dsc[i]), 2),
                                         "b": round(float(bsc[i]), 2), "g": round(float(gshare[i]), 2), "t": round(float(tree_ha[i]), 1)}})

def site_props(s):
    return {"cat": s["cat"], "name": s["name"], "c": s["ccls"], "v": round(s["cpi"], 3),
            "p": round(s["pmi"], 3), "score": s["score"], "sh": round(s["shade"], 2)}

site_feats = []
for k, s in enumerate(sites):
    g = to_wgs(arcpy.PointGeometry(arcpy.Point(s["x"], s["y"]), SR_UTM)).firstPoint
    pr = site_props(s); pr["id"] = k
    site_feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r5(g.X), r5(g.Y)]},
                       "properties": pr})
    s["lon"], s["lat"] = r5(g.X), r5(g.Y); s["id"] = k

perim_feats, fp_feats = [], []
for s, ring, kind, fpg in perims:
    perim_feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": ring_coords(to_wgs(ring))},
                        "properties": {"cat": s["cat"], "name": s["name"], "c": s["ccls"], "k": kind}})
    if fpg is not None:
        fp_feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": ring_coords(to_wgs(fpg))},
                         "properties": {"cat": s["cat"]}})

park_feats = []
with arcpy.da.SearchCursor(os.path.join(GDB, "Parks"), ["SHAPE@", "name"], spatial_reference=SR_WGS) as sc:
    for shp, nm in sc:
        park_feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": ring_coords(shp)},
                           "properties": {"name": clean(nm)}})

walk_feats = []
with arcpy.da.SearchCursor(os.path.join(GDB, "Pedestrian_Network"), ["SHAPE@", "category"], spatial_reference=SR_WGS) as sc:
    for shp, cat in sc:
        walk_feats.append({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": line_coords(shp)},
                           "properties": {"t": cat}})

corr_feats = []
for kind, p, shp, cat in corr:
    corr_feats.append({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": line_coords(to_wgs(shp))},
                       "properties": {"k": kind, "p": round(p, 3)}})

hot_feats = []
with arcpy.da.SearchCursor(os.path.join(GDB, "Pedestrian_Hotspots"), ["SHAPE@", "SHAPE@AREA"], spatial_reference=SR_WGS) as sc:
    rows = list(sc)
with arcpy.da.SearchCursor(os.path.join(GDB, "Pedestrian_Hotspots"), ["SHAPE@AREA"]) as sc:
    areas = [r[0] for r in sc]
for (shp, _), a in zip(rows, areas):
    hot_feats.append({"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": ring_coords(shp)},
                      "properties": {"a": round(a / 1e6, 2)}})
hot_feats.sort(key=lambda f: -f["properties"]["a"])

bnd_wgs = next(r[0] for r in arcpy.da.SearchCursor(BND, ["SHAPE@"], spatial_reference=SR_WGS))
bnd_feat = {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": ring_coords(bnd_wgs)}, "properties": {}}

# ranking
ranked = sorted([s for s in sites], key=lambda s: -s["score"])
def top(cat, n):
    return [{"id": s["id"], "name": s["name"] or f"Unnamed {s['cat'].lower()}", "cat": s["cat"],
             "score": s["score"], "c": s["ccls"], "lon": s["lon"], "lat": s["lat"]}
            for s in ranked if (cat is None or s["cat"] == cat)][:n]

cat_stats = {}
for cat in ["Mosque", "School", "Bus", "Mall", "Park"]:
    ss = [s for s in sites if s["cat"] == cat]
    cat_stats[cat] = {"n": len(ss), "hi": sum(1 for s in ss if s["ccls"] >= 4)}

hex_area = float(area.mean()) / 1e6
stats = {
    "city_km2": round(city_area, 0), "hex_n": int(len(oids)), "hex_km2": round(hex_area, 3),
    "cpi_counts": {str(c): int((ccls == c).sum()) for c in range(1, 6)},
    "cpi_km2": {str(c): round(float((ccls == c).sum() * hex_area), 1) for c in range(1, 6)},
    "pmi_counts": {str(c): int((pcls == c).sum()) for c in range(1, 6)},
    "cpi_breaks": [round(float(b), 3) for b in brk],
    "perim_n": len(perims), "perim_fp": sum(1 for p in perims if p[2] == "footprint"),
    "perim_km2": round(perim_area, 2), "corr_n": len(corr_feats),
    "corr_km": round(sum(c[2].length for c in corr) / 1000, 0),
    "walk_km": round(sum(r[0] for r in arcpy.da.SearchCursor(os.path.join(GDB, "Pedestrian_Network"), ["SHAPE@LENGTH"])) / 1000, 0),
    "hot_n": len(hot_feats), "hot_km2": round(sum(f["properties"]["a"] for f in hot_feats), 1),
    "cats": cat_stats, "green_hex": int((gshare > 0).sum()),
    "trees": len(tx), "buildings": int(arcpy.management.GetCount(os.path.join(GDB, "Buildings"))[0]),
    "generated": datetime.date.today().isoformat(),
}

# ---- display zones: 600 m circles around each 1 km cell centre overlap their neighbours (centres 1 km apart,
# so 577 m closes every gap); circles of the same class are dissolved; where classes overlap the higher class wins.
ZONE_R = 600.0
def zones(cls_arr, mask):
    out, higher = [], None
    for c in range(5, 0, -1):
        sel = np.where(mask & (cls_arr == c))[0]
        if not len(sel): continue
        g = None
        for i in sel:
            b = arcpy.PointGeometry(arcpy.Point(float(cx[i]), float(cy[i])), SR_UTM).buffer(ZONE_R)
            g = b if g is None else g.union(b)
        if higher is not None: g = g.difference(higher)
        higher = g if higher is None else higher.union(g)
        w = g.densify("DISTANCE", 40, 0.5).projectAs(SR_WGS)
        for poly in ring_coords(w):
            out.append({"type": "Feature", "properties": {"c": c}, "geometry": {"type": "Polygon", "coordinates": poly}})
    return out
_act = (cpi > 0) | (pmi > 0)
zcpi = zones(ccls, _act & (ccls >= 1))
_tb = np.digitize(tree_ha, [0.01, 2, 5, 10, 20])          # 1: <2, 2: 2-5, 3: 5-10, 4: 10-20, 5: 20+ trees/ha
ztree = zones(_tb, _act & (_tb >= 1))
zpmi = zones(pcls, _act & (pcls >= 2))          # Map 02: PMI zones, same method
log(f"Zones: CPI {len(zcpi)} polygons, tree density {len(ztree)} polygons")

data = {"stats": stats, "hex": {"type": "FeatureCollection", "features": hex_feats},
        "zcpi": {"type": "FeatureCollection", "features": zcpi}, "ztree": {"type": "FeatureCollection", "features": ztree},
        "zpmi": {"type": "FeatureCollection", "features": zpmi},
        "sites": {"type": "FeatureCollection", "features": site_feats},
        "perim": {"type": "FeatureCollection", "features": perim_feats},
        "fp": {"type": "FeatureCollection", "features": fp_feats},
        "parks": {"type": "FeatureCollection", "features": park_feats},
        "walk": {"type": "FeatureCollection", "features": walk_feats},
        "corr": {"type": "FeatureCollection", "features": corr_feats},
        "hot": {"type": "FeatureCollection", "features": hot_feats},
        "bnd": {"type": "FeatureCollection", "features": [bnd_feat]},
        "adm": {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": poly}} for (shp_,) in arcpy.da.SearchCursor(os.path.join(GDB, "Fallback_Boundary"), ["SHAPE@"], spatial_reference=SR_WGS) for poly in ring_coords(shp_.generalize(0.0003))]},   # administrative boundary
        "top": {"all": top(None, 25), "Mosque": top("Mosque", 10), "School": top("School", 10),
                "Bus": top("Bus", 10), "Mall": top("Mall", 10), "Park": top("Park", 10)},
        "catch": {"type": "FeatureCollection", "features": catch_feats},
        "trips": trips, "ctrips": ctrip_out, "cover": cover, "pc": pc,
        "pc_hull": {"type": "FeatureCollection", "features": pc_hull_feats},
        "pc_pts": {"type": "FeatureCollection", "features": pc_pts},
        "pcnet": pcnet,
        "umb": {"type": "FeatureCollection", "features": umb_feats},
        "trees_p": {"type": "FeatureCollection", "features": tree_feats},
        "plan": {"summary": summ, "top": plan_top, "target": TARGET, "cap": CAP_M2,
                 "tot_umb": len(umb_pts), "tot_tree": len(tree_pts) + len(street_pts), "site_tree": len(tree_pts),
                 "street": {"n": len(street_pts), "km": round(ckm), "sp": ST_SP, "by": st_by_cls}, "tiers": tiers, "crown_m2": TREE_M2, "tree_sp": TREE_SP,
                 "site": {str(r["id"]): [r["umbrellas"], r["trees"], r["suit_m2"], r["note"]] for r in plan if r.get("id") is not None},
                 "canopy_pp": round(canopy_pp, 2), "air_dT": round(air_dT, 2), "hot_km2": round(hot_area_m2 / 1e6, 1),
                 "ped_m2": round(ped_space_m2), "shade_m2": round(shade_m2),
                 "shade_pct": round(100 * shade_m2 / ped_space_m2, 0)},
        "graph": graph, "walk_speed": WALK_SPEED, "suit": suit, "stree": stree, "bld": bld_rings, "cmin": cmin, "ntime": ntime,
        "suit_w": {"heat": 30, "foot": 30, "sun": 20, "access": 20, "min": SUIT_MIN, "spacing": SPACING_U, "walk_iv": WALK_INTERVAL},
        "haram": list(HARAM)}
js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
with open(os.path.join(HERE, "madinah_data.json"), "w", encoding="utf-8") as f:
    f.write(js)
with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()
html = html.replace("/*__DATA__*/null", js)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
log(json.dumps(stats, ensure_ascii=False, indent=1))
log(f"HTML: {OUT_HTML}  ({os.path.getsize(OUT_HTML)/1e6:.1f} MB)")
arcpy.management.Delete("memory")
