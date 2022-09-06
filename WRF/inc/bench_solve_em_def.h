#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex, solve_tim
#define SOLVE_START     solve_tim = rsl_internal_microclock()
#define SOLVE_END       solve_tim = rsl_internal_microclock() - solve_tim
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex
#define BENCH_REPORT(A) write(0,*)'A= ',A
BENCH_DECL(step_prep_tim)
BENCH_DECL(set_phys_bc_tim)
BENCH_DECL(rk_tend_tim)
BENCH_DECL(relax_bdy_dry_tim)
BENCH_DECL(small_step_prep_tim)
BENCH_DECL(set_phys_bc2_tim)
BENCH_DECL(advance_uv_tim)
BENCH_DECL(spec_bdy_uv_tim)
BENCH_DECL(advance_mu_t_tim)
BENCH_DECL(spec_bdy_t_tim)
BENCH_DECL(sumflux_tim)
BENCH_DECL(advance_w_tim)
BENCH_DECL(spec_bdynhyd_tim)
BENCH_DECL(cald_p_rho_tim)
BENCH_DECL(phys_bc_tim)
BENCH_DECL(calc_mu_uv_tim)
BENCH_DECL(small_step_finish_tim)
BENCH_DECL(rk_scalar_tend_tim)
BENCH_DECL(rlx_bdy_scalar_tim)
BENCH_DECL(update_scal_tim)
BENCH_DECL(flow_depbdy_tim)
BENCH_DECL(tke_adv_tim)
BENCH_DECL(chem_adv_tim)
BENCH_DECL(calc_p_rho_tim)
BENCH_DECL(diag_w_tim)
BENCH_DECL(bc_end_tim)
BENCH_DECL(advance_ppt_tim)
BENCH_DECL(moist_physics_prep_tim)
BENCH_DECL(micro_driver_tim)
BENCH_DECL(moist_phys_end_tim)
BENCH_DECL(time_filt_tim)
BENCH_DECL(bc_2d_tim)
BENCH_DECL(microswap_1)
BENCH_DECL(microswap_2)
BENCH_DECL(tracer_adv_tim)
BENCH_DECL(rk_step_is_one_tim)
#else
#define SOLVE_START
#define SOLVE_END
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
