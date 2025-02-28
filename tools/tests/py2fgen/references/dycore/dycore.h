extern int solve_nh_run_wrapper(
    double *rho_now, int rho_now_size_0, int rho_now_size_1, double *rho_new,
    int rho_new_size_0, int rho_new_size_1, double *exner_now,
    int exner_now_size_0, int exner_now_size_1, double *exner_new,
    int exner_new_size_0, int exner_new_size_1, double *w_now, int w_now_size_0,
    int w_now_size_1, double *w_new, int w_new_size_0, int w_new_size_1,
    double *theta_v_now, int theta_v_now_size_0, int theta_v_now_size_1,
    double *theta_v_new, int theta_v_new_size_0, int theta_v_new_size_1,
    double *vn_now, int vn_now_size_0, int vn_now_size_1, double *vn_new,
    int vn_new_size_0, int vn_new_size_1, double *w_concorr_c,
    int w_concorr_c_size_0, int w_concorr_c_size_1, double *ddt_vn_apc_ntl1,
    int ddt_vn_apc_ntl1_size_0, int ddt_vn_apc_ntl1_size_1,
    double *ddt_vn_apc_ntl2, int ddt_vn_apc_ntl2_size_0,
    int ddt_vn_apc_ntl2_size_1, double *ddt_w_adv_ntl1,
    int ddt_w_adv_ntl1_size_0, int ddt_w_adv_ntl1_size_1,
    double *ddt_w_adv_ntl2, int ddt_w_adv_ntl2_size_0,
    int ddt_w_adv_ntl2_size_1, double *theta_v_ic, int theta_v_ic_size_0,
    int theta_v_ic_size_1, double *rho_ic, int rho_ic_size_0, int rho_ic_size_1,
    double *exner_pr, int exner_pr_size_0, int exner_pr_size_1,
    double *exner_dyn_incr, int exner_dyn_incr_size_0,
    int exner_dyn_incr_size_1, double *ddt_exner_phy, int ddt_exner_phy_size_0,
    int ddt_exner_phy_size_1, double *grf_tend_rho, int grf_tend_rho_size_0,
    int grf_tend_rho_size_1, double *grf_tend_thv, int grf_tend_thv_size_0,
    int grf_tend_thv_size_1, double *grf_tend_w, int grf_tend_w_size_0,
    int grf_tend_w_size_1, double *mass_fl_e, int mass_fl_e_size_0,
    int mass_fl_e_size_1, double *ddt_vn_phy, int ddt_vn_phy_size_0,
    int ddt_vn_phy_size_1, double *grf_tend_vn, int grf_tend_vn_size_0,
    int grf_tend_vn_size_1, double *vn_ie, int vn_ie_size_0, int vn_ie_size_1,
    double *vt, int vt_size_0, int vt_size_1, double *mass_flx_me,
    int mass_flx_me_size_0, int mass_flx_me_size_1, double *mass_flx_ic,
    int mass_flx_ic_size_0, int mass_flx_ic_size_1, double *vol_flx_ic,
    int vol_flx_ic_size_0, int vol_flx_ic_size_1, double *vn_traj,
    int vn_traj_size_0, int vn_traj_size_1, double dtime, int lprep_adv,
    int at_initial_timestep, double divdamp_fac_o2, double ndyn_substeps,
    int idyn_timestep);
extern int solve_nh_init_wrapper(
    double *vct_a, int vct_a_size_0, double *vct_b, int vct_b_size_0,
    double *cell_areas, int cell_areas_size_0, double *primal_normal_cell_x,
    int primal_normal_cell_x_size_0, int primal_normal_cell_x_size_1,
    double *primal_normal_cell_y, int primal_normal_cell_y_size_0,
    int primal_normal_cell_y_size_1, double *dual_normal_cell_x,
    int dual_normal_cell_x_size_0, int dual_normal_cell_x_size_1,
    double *dual_normal_cell_y, int dual_normal_cell_y_size_0,
    int dual_normal_cell_y_size_1, double *edge_areas, int edge_areas_size_0,
    double *tangent_orientation, int tangent_orientation_size_0,
    double *inverse_primal_edge_lengths, int inverse_primal_edge_lengths_size_0,
    double *inverse_dual_edge_lengths, int inverse_dual_edge_lengths_size_0,
    double *inverse_vertex_vertex_lengths,
    int inverse_vertex_vertex_lengths_size_0, double *primal_normal_vert_x,
    int primal_normal_vert_x_size_0, int primal_normal_vert_x_size_1,
    double *primal_normal_vert_y, int primal_normal_vert_y_size_0,
    int primal_normal_vert_y_size_1, double *dual_normal_vert_x,
    int dual_normal_vert_x_size_0, int dual_normal_vert_x_size_1,
    double *dual_normal_vert_y, int dual_normal_vert_y_size_0,
    int dual_normal_vert_y_size_1, double *f_e, int f_e_size_0, double *c_lin_e,
    int c_lin_e_size_0, int c_lin_e_size_1, double *c_intp, int c_intp_size_0,
    int c_intp_size_1, double *e_flx_avg, int e_flx_avg_size_0,
    int e_flx_avg_size_1, double *geofac_grdiv, int geofac_grdiv_size_0,
    int geofac_grdiv_size_1, double *geofac_rot, int geofac_rot_size_0,
    int geofac_rot_size_1, double *pos_on_tplane_e_1,
    int pos_on_tplane_e_1_size_0, int pos_on_tplane_e_1_size_1,
    double *pos_on_tplane_e_2, int pos_on_tplane_e_2_size_0,
    int pos_on_tplane_e_2_size_1, double *rbf_vec_coeff_e,
    int rbf_vec_coeff_e_size_0, int rbf_vec_coeff_e_size_1, double *e_bln_c_s,
    int e_bln_c_s_size_0, int e_bln_c_s_size_1, double *rbf_coeff_1,
    int rbf_coeff_1_size_0, int rbf_coeff_1_size_1, double *rbf_coeff_2,
    int rbf_coeff_2_size_0, int rbf_coeff_2_size_1, double *geofac_div,
    int geofac_div_size_0, int geofac_div_size_1, double *geofac_n2s,
    int geofac_n2s_size_0, int geofac_n2s_size_1, double *geofac_grg_x,
    int geofac_grg_x_size_0, int geofac_grg_x_size_1, double *geofac_grg_y,
    int geofac_grg_y_size_0, int geofac_grg_y_size_1, double *nudgecoeff_e,
    int nudgecoeff_e_size_0, int *bdy_halo_c, int bdy_halo_c_size_0,
    int *mask_prog_halo_c, int mask_prog_halo_c_size_0, double *rayleigh_w,
    int rayleigh_w_size_0, double *exner_exfac, int exner_exfac_size_0,
    int exner_exfac_size_1, double *exner_ref_mc, int exner_ref_mc_size_0,
    int exner_ref_mc_size_1, double *wgtfac_c, int wgtfac_c_size_0,
    int wgtfac_c_size_1, double *wgtfacq_c, int wgtfacq_c_size_0,
    int wgtfacq_c_size_1, double *inv_ddqz_z_full, int inv_ddqz_z_full_size_0,
    int inv_ddqz_z_full_size_1, double *rho_ref_mc, int rho_ref_mc_size_0,
    int rho_ref_mc_size_1, double *theta_ref_mc, int theta_ref_mc_size_0,
    int theta_ref_mc_size_1, double *vwind_expl_wgt, int vwind_expl_wgt_size_0,
    double *d_exner_dz_ref_ic, int d_exner_dz_ref_ic_size_0,
    int d_exner_dz_ref_ic_size_1, double *ddqz_z_half, int ddqz_z_half_size_0,
    int ddqz_z_half_size_1, double *theta_ref_ic, int theta_ref_ic_size_0,
    int theta_ref_ic_size_1, double *d2dexdz2_fac1_mc,
    int d2dexdz2_fac1_mc_size_0, int d2dexdz2_fac1_mc_size_1,
    double *d2dexdz2_fac2_mc, int d2dexdz2_fac2_mc_size_0,
    int d2dexdz2_fac2_mc_size_1, double *rho_ref_me, int rho_ref_me_size_0,
    int rho_ref_me_size_1, double *theta_ref_me, int theta_ref_me_size_0,
    int theta_ref_me_size_1, double *ddxn_z_full, int ddxn_z_full_size_0,
    int ddxn_z_full_size_1, double *zdiff_gradp, int zdiff_gradp_size_0,
    int zdiff_gradp_size_1, int zdiff_gradp_size_2, int *vertoffset_gradp,
    int vertoffset_gradp_size_0, int vertoffset_gradp_size_1,
    int vertoffset_gradp_size_2, int *ipeidx_dsl, int ipeidx_dsl_size_0,
    int ipeidx_dsl_size_1, double *pg_exdist, int pg_exdist_size_0,
    int pg_exdist_size_1, double *ddqz_z_full_e, int ddqz_z_full_e_size_0,
    int ddqz_z_full_e_size_1, double *ddxt_z_full, int ddxt_z_full_size_0,
    int ddxt_z_full_size_1, double *wgtfac_e, int wgtfac_e_size_0,
    int wgtfac_e_size_1, double *wgtfacq_e, int wgtfacq_e_size_0,
    int wgtfacq_e_size_1, double *vwind_impl_wgt, int vwind_impl_wgt_size_0,
    double *hmask_dd3d, int hmask_dd3d_size_0, double *scalfac_dd3d,
    int scalfac_dd3d_size_0, double *coeff1_dwdz, int coeff1_dwdz_size_0,
    int coeff1_dwdz_size_1, double *coeff2_dwdz, int coeff2_dwdz_size_0,
    int coeff2_dwdz_size_1, double *coeff_gradekin, int coeff_gradekin_size_0,
    int coeff_gradekin_size_1, int *c_owner_mask, int c_owner_mask_size_0,
    double *cell_center_lat, int cell_center_lat_size_0,
    double *cell_center_lon, int cell_center_lon_size_0,
    double *edge_center_lat, int edge_center_lat_size_0,
    double *edge_center_lon, int edge_center_lon_size_0,
    double *primal_normal_x, int primal_normal_x_size_0,
    double *primal_normal_y, int primal_normal_y_size_0,
    double rayleigh_damping_height, int itime_scheme, int iadv_rhotheta,
    int igradp_method, double ndyn_substeps, int rayleigh_type,
    double rayleigh_coeff, int divdamp_order, int is_iau_active,
    double iau_wgt_dyn, int divdamp_type, double divdamp_trans_start,
    double divdamp_trans_end, int l_vert_nested, double rhotheta_offctr,
    double veladv_offctr, double max_nudging_coeff, double divdamp_fac,
    double divdamp_fac2, double divdamp_fac3, double divdamp_fac4,
    double divdamp_z, double divdamp_z2, double divdamp_z3, double divdamp_z4,
    double lowest_layer_thickness, double model_top_height,
    double stretch_factor, double mean_cell_area, int nflat_gradp,
    int num_levels);