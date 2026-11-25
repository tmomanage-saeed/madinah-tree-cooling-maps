# Builds a 1 km grid (circles of 500 m radius from the centre; hexagon cells 86.6 ha, 1 km centre to centre) from the 250 m (5 ha) Pedestrian_Movement_Hex by area-weighted averaging.
# Aggregation only (fine -> coarse); no new values are invented. PMI classes use the same PMI breaks as the 250 m grid.
import arcpy, os, numpy as np
arcpy.env.overwriteOutput = True
ROOT = r"C:\GIS\Madinah_Map"
SRC = os.path.join(ROOT, "Madinah_City.gdb", "Pedestrian_Movement_Hex")
AGDB = os.path.join(ROOT, "Madinah_Analysis.gdb")
OUT = os.path.join(AGDB, "Pedestrian_Movement_Cell_1km")
CELL_M2 = 2 * 3 ** 0.5 * 500 ** 2     # hexagon with 500 m inradius = 866,025 m2
BRK = [0.048, 0.096, 0.163, 0.244]
LBL = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
FIELDS = ["Walk_Score", "Dest_Score", "Built_Score", "PMI"]
labels = sorted({r[0] for r in arcpy.da.SearchCursor(SRC, ["PMI_Label", "PMI_Class"])}, key=str)
lab_by_cls = {c: l for l, c in arcpy.da.SearchCursor(SRC, ["PMI_Label", "PMI_Class"])}
ext = arcpy.Describe(SRC).extent
tess = arcpy.management.GenerateTessellation(os.path.join("memory", "tess"), ext, "HEXAGON", f"{CELL_M2:.0f} SquareMeters", arcpy.Describe(SRC).spatialReference)[0]
ix = arcpy.analysis.Intersect([[tess, ""], [SRC, ""]], os.path.join("memory", "hix"), "ALL")[0]
fid = next(f.name for f in arcpy.ListFields(ix) if f.name.upper() == "FID_TESS")
acc = {}
with arcpy.da.SearchCursor(ix, [fid, "SHAPE@AREA"] + FIELDS) as sc:
    for r in sc:
        a = acc.setdefault(r[0], np.zeros(len(FIELDS) + 1)); w = r[1]
        a[0] += w; a[1:] += w * np.array([v or 0 for v in r[2:]])
keep = [k for k, a in acc.items() if a[0] >= 0.25 * CELL_M2]      # at least 25 % covered by the source grid
lyr = arcpy.management.MakeFeatureLayer(tess, "tess_lyr", f"OBJECTID IN ({','.join(map(str, keep))})")[0]
arcpy.management.CopyFeatures(lyr, OUT)
for f in FIELDS: arcpy.management.AddField(OUT, f, "DOUBLE")
arcpy.management.AddField(OUT, "PMI_Class", "SHORT"); arcpy.management.AddField(OUT, "PMI_Label", "TEXT")
arcpy.management.AddField(OUT, "Src_Cover", "DOUBLE")
# OBJECTID order is preserved by CopyFeatures of a filtered layer -> map by GRID_ID
idmap = {}
with arcpy.da.SearchCursor(lyr, ["OBJECTID", "GRID_ID"]) as sc:
    for o, g in sc: idmap[g] = o
n = {c: 0 for c in range(1, 6)}
with arcpy.da.UpdateCursor(OUT, ["GRID_ID"] + FIELDS + ["PMI_Class", "PMI_Label", "Src_Cover"]) as uc:
    for r in uc:
        a = acc[idmap[r[0]]]; v = a[1:] / a[0]
        c = int(np.digitize(v[3], BRK) + 1); n[c] += 1
        uc.updateRow([r[0]] + [round(float(x), 4) for x in v] + [c, lab_by_cls.get(c, LBL[c]), round(float(a[0] / CELL_M2), 3)])
print("1 km cells:", int(arcpy.management.GetCount(OUT)[0]), "classes", n, "labels", lab_by_cls)
