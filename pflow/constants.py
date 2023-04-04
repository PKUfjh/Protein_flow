# rid task name format
argo_namespace = "argo"
walker_tag_fmt = "{idx:03d}"
init_conf_name = "conf_{idx:03d}.gro"
init_input_name = "input_{idx:03d}.lammps"
explore_task_pattern = "{:03d}"
explore_task_file = "explore_{walker:03d}.pkl"
cluster_selection_data_name = "cls_sel.out.npy"
cluster_selection_index_name = "cls_sel.ndx.npy"
sel_ndx_name = "sel.ndx"
cv_init_label = "cv_init_{walker:03d}_{idx:d}.out"
model_devi_name = "model_devi.txt"
label_task_pattern = "{:03d}"
cv_force_out = "cv_forces.out"
data_new = "data.new.npy"
data_old = "data.old.npy"
data_raw = "data.raw.npy"
block_tag_fmt = "iter-{idx_iter:03d}"

# PLUMED2 file names
plumed_input_name = "plumed.dat"
plumed_bf_input_name = "plumed_bf.dat"
plumed_restraint_input_name = "plumed_restraint.dat"
plumed_output_name = "plm.out"
center_out_name = "centers.out"

# Gromacs file names
gmx_conf_name = "conf.gro"
gmx_top_name = "topol.top"
gmx_idx_name = "index.ndx"
gmx_mdp_name = "grompp.mdp"
gmx_tpr_name = "topol.tpr"
gmx_grompp_log = "gmx_grompp.log"
gmx_mdrun_log = "md.log"
restraint_md_mdp_name = "grompp_restraint.mdp"
gmx_trr_name = "traj.trr"
gmx_xtc_name = "traj_comp.xtc"
sel_gro_name = "conf_{walker:03d}_{idx:d}.gro"
sel_gro_name_gmx = "conf_.gro"
gmx_conf_out = "confout.gro"
gmx_align_name = "traj_aligned_pro.xtc"

# npz file
traj_npz_name = 'traj_aligned.npz'
combined_npz_name = "combined_traj.npz"

# Model files
model_tag_fmt = "{idx:03d}"

# Units
kb = 8.617333E-5
kbT = (8.617333E-5) * 300
beta = 1.0 / kbT
f_cvt = 96.485
inverse_f_cvt = 1 / f_cvt

# precision
model_devi_precision = "%.6e"