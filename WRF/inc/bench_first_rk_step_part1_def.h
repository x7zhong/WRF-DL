#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex, first_rk_step_part1_tim
#define FIRST_PART1_START     first_rk_step_part1_tim = rsl_internal_microclock()
#define FIRST_PART1_END       first_rk_step_part1_tim = rsl_internal_microclock() - first_rk_step_part1_tim   
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex
#define BENCH_REPORT(A) write(0,*)'A= ',A
BENCH_DECL(init_zero_tend_tim)
BENCH_DECL(phy_prep_tim)
BENCH_DECL(rad_driver_tim)
BENCH_DECL(fire_driver_tim)
BENCH_DECL(surf_driver_tim)
BENCH_DECL(pbl_driver_tim)
BENCH_DECL(cu_driver_tim)
BENCH_DECL(shcu_driver_tim)
BENCH_DECL(fdda_driver_tim)
BENCH_DECL(pre_radiation_driver_tim)
BENCH_DECL(force_scm_tim)
#else
#define FIRST_PART1_START  
#define FIRST_PART1_END   
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
