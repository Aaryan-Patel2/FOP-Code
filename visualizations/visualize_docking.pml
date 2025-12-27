# PyMOL visualization script for docked ligands with color coding
# Run in PyMOL: File > Run Script > visualize_docking.pml

# Load receptor
load receptor_siteA.pdbqt, receptor
hide everything, receptor
show surface, receptor
color white, receptor
set transparency, 0.5, receptor
set surface_quality, 1

# Load reference ligand (BASELINE)
load reference_A_docked.pdbqt, reference_A
hide everything, reference_A
show sticks, reference_A
color lightpink, reference_A
set stick_radius, 0.15, reference_A

# Load generated ligands with color coding by affinity
# ligand_4: -12.85 kcal/mol (BEST - Gold)
load ligand_4_docked.pdbqt, ligand_4
hide everything, ligand_4
show sticks, ligand_4
util.cbay ligand_4
set stick_radius, 0.20, ligand_4

# ligand_2: -10.17 kcal/mol (2nd - Cyan)
load ligand_2_docked.pdbqt, ligand_2
hide everything, ligand_2
show sticks, ligand_2
util.cbac ligand_2
set stick_radius, 0.18, ligand_2

# ligand_1: -10.06 kcal/mol (3rd - Green)
load ligand_1_docked.pdbqt, ligand_1
hide everything, ligand_1
show sticks, ligand_1
util.cbag ligand_1
set stick_radius, 0.18, ligand_1

# ligand_3: -9.57 kcal/mol (4th - Blue)
load ligand_3_docked.pdbqt, ligand_3
hide everything, ligand_3
show sticks, ligand_3
util.cbab ligand_3
set stick_radius, 0.16, ligand_3

# ligand_5: -9.43 kcal/mol (5th - Magenta)
load ligand_5_docked.pdbqt, ligand_5
hide everything, ligand_5
show sticks, ligand_5
util.cbam ligand_5
set stick_radius, 0.16, ligand_5

# Zoom to binding site
zoom ligand_4, 20

# Set background and display
bg_color white
set antialias, 2
set ray_shadows, 1
set depth_cue, 0
set specular, 0.5

# Remove labels
# label ligand_4 and name C1, "Best: -12.85"
# label reference_A and name C1, "Ref: -9.13"

# Orient view
orient ligand_4

# Legend info
print("============================================================")
print("LIGAND COLOR CODING:")
print("============================================================")
print("GOLD   (ligand_4):   -12.85 kcal/mol (41 percent better)")
print("CYAN   (ligand_2):   -10.17 kcal/mol (11 percent better)")
print("GREEN  (ligand_1):   -10.06 kcal/mol (10 percent better)")
print("BLUE   (ligand_3):    -9.57 kcal/mol (5 percent better)")
print("MAGENTA(ligand_5):    -9.43 kcal/mol (3 percent better)")
print("WHITE  (reference_A): -9.13 kcal/mol (BASELINE)")
print("============================================================")
print("RED SPHERE: Binding site center (24.87, -12.54, 38.40)")
print("============================================================")

# Save session
save docking_visualization.pse
print("Session saved: docking_visualization.pse")

# Optional: render high-quality image
# ray 2400, 1800
# png docking_comparison.png, dpi=300
