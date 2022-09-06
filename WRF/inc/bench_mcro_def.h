#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex_int
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex_int=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex_int
#define BENCH_REPORT(A) write(0,*)'A= ',A
#else
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
