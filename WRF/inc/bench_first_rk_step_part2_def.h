#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex, first_rk_step_part2_tim
#define FIRST_PART2_START     first_rk_step_part2_tim = rsl_internal_microclock()
#define FIRST_PART2_END       first_rk_step_part2_tim = rsl_internal_microclock() - first_rk_step_part2_tim
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex
#define BENCH_REPORT(A) write(0,*)'A= ',A
BENCH_DECL(comp_diff_metrics_tim)
BENCH_DECL(tke_diff_bc_tim)
BENCH_DECL(deform_div_tim)
BENCH_DECL(calc_tke_tim)
BENCH_DECL(phy_bc_tim)
BENCH_DECL(update_phy_ten_tim)
BENCH_DECL(tke_rhs_tim)
BENCH_DECL(vert_diff_tim)
BENCH_DECL(hor_diff_tim)
BENCH_DECL(cal_phy_tend)
BENCH_DECL(helicity_tim)
#else
#define FIRST_PART2_START
#define FIRST_PART2_END
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
