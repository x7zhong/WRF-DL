#ifdef BENCH
   INTEGER, EXTERNAL :: rsl_internal_microclock
   INTEGER btimex
#define BENCH_DECL(A)   integer A
#define BENCH_INIT(A)   A=0
#define BENCH_START(A)  btimex=rsl_internal_microclock()
#define BENCH_END(A)    A=A+rsl_internal_microclock()-btimex
#define BENCH_REPORT(A) write(0,*)'A= ',A
BENCH_DECL(cpl_rcv_tim)
BENCH_DECL(rainbl_tim)
BENCH_DECL(sst_skin_update_tim)
BENCH_DECL(tmnupdate_tim)
BENCH_DECL(topo_rad_adj_drvr_tim)
BENCH_DECL(sfclay_select_tim)
BENCH_DECL(sf_fogdes_tim)
BENCH_DECL(ocean_driver_tim)
BENCH_DECL(lake_tim)
BENCH_DECL(sfc_select_clmdrv_tim)
BENCH_DECL(sfc_select_sfcdiags_tim)
#else
#define BENCH_INIT(A)
#define BENCH_START(A)
#define BENCH_END(A)
#define BENCH_REPORT(A)
#endif
