# PyMOL visualization script for ligand_2 docking
# Run in PyMOL: File > Run Script > visualize_ligand_2.pml

# Load receptor
load receptor_siteA.pdbqt, receptor
hide everything, receptor
show surface, receptor
color white, receptor
set transparency, 0.5, receptor
set surface_quality, 1

# Load ligand_2
load ligand_2_docked.pdbqt, ligand_2
hide everything, ligand_2
show sticks, ligand_2
color firebrick, ligand_2
set stick_radius, 0.25, ligand_2

# Load ligand_2
load ligand_2_docked.pdbqt, ligand_2
hide everything, ligand_2
show sticks, ligand_2
util.cbac ligand_2
set stick_radius, 0.18, ligand_2

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
zoom ligand_2, 8

# Set background and display
bg_color white
set antialias, 2
set ray_shadows, 1
set depth_cue, 0
set specular, 0.5
set ray_trace_mode, 1

# Orient view
orient ligand_2

# Save session
save ligand_2_visualization.pse
print("Session saved: ligand_2_visualization.pse")

# Optional: render high-quality image
# ray 1200, 1200
# png ligand_2_docking.png, dpi=300
