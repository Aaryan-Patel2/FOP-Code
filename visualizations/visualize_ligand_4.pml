# PyMOL visualization script for ligand_4 docking
# Run in PyMOL: File > Run Script > visualize_ligand_4.pml

# Load receptor
load receptor_siteA.pdbqt, receptor
hide everything, receptor
show surface, receptor
color white, receptor
set transparency, 0.5, receptor
set surface_quality, 1

# Load ligand_4
load ligand_4_docked.pdbqt, ligand_4
hide everything, ligand_4
show sticks, ligand_4
color firebrick, ligand_4
set stick_radius, 0.25, ligand_4

# Load ligand_4
load ligand_4_docked.pdbqt, ligand_4
hide everything, ligand_4
show sticks, ligand_4
util.cbay ligand_4
set stick_radius, 0.20, ligand_4

# Add reference points for consistent alignment
pseudoatom ref1, pos=[24.87, -12.54, 38.40]
pseudoatom ref2, pos=[20.0, -15.0, 35.0]
pseudoatom ref3, pos=[28.0, -10.0, 40.0]
hide everything, ref1 ref2 ref3
show spheres, ref1 ref2 ref3
color gray50, ref1 ref2 ref3
set sphere_scale, 0.3, ref1 ref2 ref3
set sphere_transparency, 0.7, ref1 ref2 ref3

# Zoom to binding site
zoom ligand_4, 8

# Set background and display
bg_color white
set antialias, 2
set ray_shadows, 1
set depth_cue, 0
set specular, 0.5
set ray_trace_mode, 1

# Orient view
orient ligand_4

# Save session
save ligand_4_visualization.pse
print("Session saved: ligand_4_visualization.pse")

# Optional: render high-quality image
# ray 1200, 1200
# png ligand_4_docking.png, dpi=300
