# Madinah: Cooling Priority & Street Trees (Tasks 5 & 6), v3

This repo holds two interactive web maps built from open data: OpenStreetMap / Geofabrik, the Meta/WRI 1 m canopy height map, and a walking network model.

**Open:** download `web/Madinah_Cooling_Pedestrian_Maps.html` and open it in Chrome. The basemap (Esri World Imagery) needs an internet connection.

## Map 01: Priority Areas for Cooling (Task 5)
- **Grid:** 1 km cells, each a 500 m radius around its centre (843 cells).
  - Built by area-weighted aggregation of the 250 m pedestrian movement grid (`make_cells_1km.py`).
  - Shown as overlapping circles, dissolved per priority class.
- **Cooling Priority Index:** 40 % sensitive uses + 35 % pedestrian movement + 25 % shade deficit.
- **Tree crown:** 53 m² (about 8 m across), the mean of 40,800 single trees measured in Madinah on the Meta/WRI 1 m canopy height map (Tolan et al. 2024).
- **Spacing by priority:** crown ÷ target kerb shade, on both kerbs of every walkable street.

| Priority | Critical | Very High | High | Moderate | Low |
|---|---|---|---|---|---|
| Spacing | 8 m | 10 m | 12 m | 16 m | 24 m |

- **Result:** 623,685 trees (611,871 street + 11,814 at sites). Air in High+ zones about 0.36 °C cooler when the trees are mature (0.33 °C per +10 % canopy).
- **Click anywhere** for the full priority card: class, rank, score components, trees in the cell, spacing and cooling.
- **"Trees per hectare" view** shows the planting density at city scale.

## Map 02: Pedestrian Movement (Task 6)
- Pedestrian Movement Index, animated corridors.
- 700 pedestrians walk to Masjid an-Nabawi and 4,989 walk across the city.
- 5/10/15-min walking catchments.
- A→B tool: fastest vs. cool route, 🌳 shade stops, and a route tree plan (tree every 8 m).
- Validated against PedestrianCatch: median difference 0.79 min, 93.8 % within ±2 min.

## Files
| File | Purpose |
|---|---|
| `web/Madinah_Cooling_Pedestrian_Maps.html` | Final deliverable (data embedded) |
| `web/template.html` | Map UI (MapLibre GL) |
| `web/build_web_data.py` | ArcGIS Pro (arcpy) pipeline |
| `web/make_cells_1km.py` | 1 km cell grid from the 250 m grid |
| `web/inject.py` | Puts the data into the template |
| `Action_Requirement_Plan.csv` | Site-by-site table |
