#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex, first_rk_step_part1_tim
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex
#define BENCH_REPORT(A) write(0,*)'A= ',A
BENCH_DECL(lw_sw_preprocess_tim)
BENCH_DECL(infer_run_tim)
BENCH_DECL(copy_tim)
#else
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
