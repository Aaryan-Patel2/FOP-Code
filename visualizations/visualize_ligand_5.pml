# PyMOL visualization script for ligand_5 docking
# Run in PyMOL: File > Run Script > visualize_ligand_5.pml

# Load receptor
load receptor_siteA.pdbqt, receptor
hide everything, receptor
show surface, receptor
color white, receptor
set transparency, 0.5, receptor
set surface_quality, 1

# Load ligand_5
load ligand_5_docked.pdbqt, ligand_5
hide everything, ligand_5
show sticks, ligand_5
color firebrick, ligand_5
set stick_radius, 0.25, ligand_5

# Load ligand_5
load ligand_5_docked.pdbqt, ligand_5
hide everything, ligand_5
show sticks, ligand_5
util.cbam ligand_5
set stick_radius, 0.16, ligand_5

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
zoom ligand_5, 8

# Set background and display
bg_color white
set antialias, 2
set ray_shadows, 1
set depth_cue, 0
set specular, 0.5
set ray_trace_mode, 1

# Orient view
orient ligand_5

# Save session
save ligand_5_visualization.pse
print("Session saved: ligand_5_visualization.pse")

# Optional: render high-quality image
# ray 1200, 1200
# png ligand_5_docking.png, dpi=300
