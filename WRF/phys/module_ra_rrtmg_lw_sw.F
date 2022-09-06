!------------------------------------------------------------------

!#define BENCH_START(A)
!#define BENCH_END(A)
MODULE module_ra_rrtmg_lw_sw

USE module_wrf_error
!by xiaohui and lifei
USE module_ra_rrtmg_lw_sw_dl_replace
USE module_wrf_infer,only: wrf_infer_run 

#if (HWRF == 1)
   USE module_state_description, ONLY : FER_MP_HIRES, FER_MP_HIRES_ADVECT, ETAMP_HWRF 
#else
   USE module_state_description, ONLY : FER_MP_HIRES, FER_MP_HIRES_ADVECT
#endif

    real, PARAMETER :: deltap = 4.  ! Pressure interval for buffer layer in mb

CONTAINS
!------------------------------------------------------------------

    subroutine inference(                                         &
                       nlayers, emiss,                            &
                       p8w, p3d, pi,                              &
                       tsk, t3d, t8w, r, g,                       &
                       icloud, warm_rain, cldfra3d,               &
                       f_ice_phy, f_rain_phy,                     &
                       xland, xice, snow,                         &
                       qv3d, qc3d, qr3d,                          &
                       qi3d, qs3d, qg3d,                          &
                       o3input, o33d,                             &
                       f_qv, f_qc, f_qr, f_qi, f_qs, f_qg,        &
                       has_reqc, has_reqi, has_reqs,              &  ! G. Thompson
!ccc added for time varying gases.
                       yr,julian,                                 &
!ccc
                       mp_physics,                                &
                       ids,ide, jds,jde, kds,kde,                 & 
                       ims,ime, jms,jme, kms,kme,                 &
                       its,ite, jts,jte, kts,kte,                 &
                       solcon, obscur, albedo, coszen,            &
                       RTHRATEN, RTHRATENLW, RTHRATENSW, GLW, GSW  )    

   IMPLICIT NONE
!------------------------------------------------------------------

   INTEGER, INTENT(IN )      ::        ids,ide, jds,jde, kds,kde, &
                                       ims,ime, jms,jme, kms,kme, &
                                       its,ite, jts,jte, kts,kte

   INTEGER, INTENT(IN )      ::        nlayers         ! total number of layers                                       

   LOGICAL, INTENT(IN )      ::        warm_rain

   INTEGER, INTENT(IN )      ::        ICLOUD
   INTEGER, INTENT(IN )      ::        MP_PHYSICS
!
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         INTENT(IN   ) ::                                    t3d, &
                                                             t8w, &
                                                             p8w, &
                                                             p3d, &
                                                             pi

   REAL, DIMENSION( ims:ime, jms:jme )                          , &
         INTENT(IN   )  ::                                  EMISS, &
                                                             TSK, &
                                                             ALBEDO, &
                                                             COSZEN, &
                                                             obscur                                                             

   REAL, INTENT(IN  )   ::                                   R,G

   REAL, DIMENSION( ims:ime, jms:jme )                          , &
         INTENT(IN   )  ::                                 XLAND, &
                                                            XICE, &
                                                            SNOW
!ccc Added for time-varying trace gases.
   INTEGER, INTENT(IN    ) ::                                 yr
   REAL, INTENT(IN    ) ::                                julian
!ccc

!
! Optional
!
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) ::                                         &
                                                        CLDFRA3D, &
                                                            QV3D, &
                                                            QC3D, &
                                                            QR3D, &
                                                            QI3D, &
                                                            QS3D, &
                                                            QG3D
   INTEGER, INTENT(IN):: has_reqc, has_reqi, has_reqs

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) ::                                         &
                                                       F_ICE_PHY, &
                                                      F_RAIN_PHY

   LOGICAL, OPTIONAL, INTENT(IN)   ::                             &
                                   F_QV,F_QC,F_QR,F_QI,F_QS,F_QG

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme ),                  &
         INTENT(INOUT)  ::                              RTHRATEN, &
                                                      RTHRATENLW, &
                                                      RTHRATENSW

   REAL, DIMENSION( ims:ime, jms:jme ),                  &
         INTENT(INOUT)  ::                              GLW, &
                                                        GSW                                                      

!  Ozone
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) :: O33D
   INTEGER, OPTIONAL, INTENT(IN ) :: o3input
 
!by xiaohui for training nn
    real, dimension( kms:kme-1, its:ite, jts:jte ) ::                tlay_3d, &
                                                                     play_3d, &
                                                                     cldfrac_3d, &
                                                                     o3vmr_3d, &
                                                                     qv_3d, &
                                                                     qc_3d, &
                                                                     qr_3d, &
                                                                     qi_3d, &
                                                                     qs_3d, &
                                                                     qg_3d                                                                  

    real, dimension( kms:kme, its:ite, jts:jte ) ::                tlev_3d, &
                                                                   plev_3d, &
                                                                   pi_3d, &
                                                                   rthraten_3d, &
                                                                   rthratenlw_3d, &
                                                                   rthratensw_3d, &
                                                                   lwuflx_3d, &
                                                                   lwdflx_3d, &
                                                                   swuflx_3d, &
                                                                   swdflx_3d

    real, dimension( its:ite, jts:jte ) ::                landfrac_2d, &
                                                          icefrac_2d, &
                                                          snow_2d, &
                                                          tsfc_2d, &
                                                          emis_2d, &
                                                          solcon_2d, &
                                                          albedo_2d, &
                                                          coszen_2d

    REAL, INTENT(IN    )      ::                             SOLCON      

    INTEGER :: i,j,k

! Define benchmarking timers if -DBENCH is compiled
#include "bench_ra_rrtmg_lw_sw_def.h"
#include "bench_ra_rrtmg_lw_sw_init.h"

       CALL wrf_debug(100, 'Call rrtmg_lw_sw_preprocess')

BENCH_START(lw_sw_preprocess_tim)
       CALL rrtmg_lw_sw_preprocess(nlayers=nlayers, emiss=EMISS,                     &
                 p8w=p8w, p3d=p3d, pi=pi,                           &
                 tsk=tsk, t3d=t3d, t8w=t8w, r=r, g=G,          &
                 icloud=icloud, warm_rain=warm_rain, cldfra3d=cldfra3d,               &
                 f_ice_phy=f_ice_phy, f_rain_phy=f_rain_phy, &
                 xland=xland, xice=xice, snow=snow,                         &
                 qv3d=qv3d, qc3d=qc3d, qr3d=qr3d,                          &
                 qi3d=qi3d, qs3d=qs3d, qg3d=qg3d,                          &
                 o3input=o3input, o33d=o33d,                             &
                 f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, &
                 f_qi=f_qi, f_qs=f_qs, f_qg=f_qg,        &
                 has_reqc=has_reqc, has_reqi=has_reqi, has_reqs=has_reqs,              &  ! G. Thompson
!ccc added for time varying gases.
                 yr=yr,julian=julian,                                 &
!ccc
                 mp_physics=mp_physics,                                &
                 IDS=ids,IDE=ide, JDS=jds,JDE=jde, KDS=kds,KDE=kde,&
                 IMS=ims,IME=ime, JMS=jms,JME=jme, KMS=kms,KME=kme,&
                 ITS=its,ITE=ite, JTS=jts,JTE=jte, KTS=kts,KTE=kte,&
                 tlay_3d=tlay_3d, tlev_3d=tlev_3d, play_3d=play_3d, &
                 plev_3d=plev_3d, pi_3d=pi_3d, cldfrac_3d=cldfrac_3d, &
                 o3vmr_3d=o3vmr_3d, qv_3d=qv_3d, qc_3d=qc_3d, qr_3d=qr_3d, &
                 qi_3d=qi_3d, qs_3d=qs_3d, qg_3d=qg_3d, &
                 landfrac_2d=landfrac_2d, icefrac_2d=icefrac_2d, &
                 snow_2d=snow_2d, tsfc_2d=tsfc_2d, emis_2d=emis_2d, &
                 solcon_2d=solcon_2d, albedo_2d=albedo_2d, coszen_2d=coszen_2d, &
                 solcon=solcon, obscur=obscur, albedo=ALBEDO, coszen=coszen   )

BENCH_END(lw_sw_preprocess_tim)

     CALL wrf_debug(100, 'Call infer_run')

BENCH_START(infer_run_tim)
    CALL wrf_infer_run(emis_2d, ite-its+1, jte-jts+1, kme-1-kms+1, kme-kms+1, &
                     solcon_2d, &
                     albedo_2d, &
                     landfrac_2d, &   
                     icefrac_2d,  &
                     snow_2d,     &
                     coszen_2d,      &       
                     tsfc_2d, &
                     tlay_3d,  &
                     tlev_3d, &
                     play_3d,  &
                     plev_3d,  &
                     qv_3d, &
                     qc_3d, &
                     qr_3d, &
                     qi_3d,  &
                     qs_3d, &
                     qg_3d,  &
                     o3vmr_3d,  &
                     cldfrac_3d, &
                     pi_3d, &
                     rthraten_3d, &
                     rthratenlw_3d, &
                     rthratensw_3d, &
                     lwuflx_3d, &
                     lwdflx_3d, &
                     swuflx_3d, &
                     swdflx_3d)   
BENCH_END(infer_run_tim)


     CALL wrf_debug(100, 'Copy inference data')

BENCH_START(copy_tim)

             DO j=jts,jte
             DO i=its,ite 
                 GLW(i,j) = lwdflx_3d(1,i,j)
                 GSW(i,j) = swdflx_3d(1,i,j) - swuflx_3d(1,i,j)
             ENDDO
             ENDDO

             DO j=jts,jte
             DO k=kts,kte
             DO i=its,ite                
                RTHRATEN(I,K,J)=rthraten_3d(K,I,J)
                RTHRATENLW(I,K,J)=rthratenlw_3d(K,I,J)
                RTHRATENSW(I,K,J)=rthratensw_3d(K,I,J)

             ENDDO
             ENDDO
             ENDDO
BENCH_END(copy_tim)

#include "bench_ra_rrtmg_lw_sw_end.h"

    end subroutine inference

   SUBROUTINE rrtmg_lw_sw_preprocess(                             &
                       nlayers, emiss,                            &
                       p8w, p3d, pi,                              &
                       tsk, t3d, t8w, r, g,                       &
                       icloud, warm_rain, cldfra3d,               &
                       f_ice_phy, f_rain_phy,                     &
                       xland, xice, snow,                         &
                       qv3d, qc3d, qr3d,                          &
                       qi3d, qs3d, qg3d,                          &
                       o3input, o33d,                             &
                       f_qv, f_qc, f_qr, f_qi, f_qs, f_qg,        &
                       has_reqc, has_reqi, has_reqs,              &  ! G. Thompson
!ccc added for time varying gases.
                       yr,julian,                                 &
!ccc
                       mp_physics,                                &
                       ids,ide, jds,jde, kds,kde,                 & 
                       ims,ime, jms,jme, kms,kme,                 &
                       its,ite, jts,jte, kts,kte,                 &
                       tlay_3d, tlev_3d, play_3d, plev_3d, pi_3d, cldfrac_3d, &
                       o3vmr_3d, qv_3d, qc_3d, qr_3d, qi_3d, qs_3d, qg_3d, &
                       landfrac_2d, icefrac_2d, snow_2d, tsfc_2d, emis_2d, &
                       solcon_2d, albedo_2d, coszen_2d, &
                       solcon, obscur, albedo, coszen              )

!xiaohui

!------------------------------------------------------------------
!ccc To use clWRF time varying trace gases
   USE MODULE_RA_CLWRF_SUPPORT, ONLY : read_CAMgases

   USE module_ra_rrtmg_lw   , ONLY : INIRAD
!

   IMPLICIT NONE
!------------------------------------------------------------------

   INTEGER, INTENT(IN )      ::        ids,ide, jds,jde, kds,kde, &
                                       ims,ime, jms,jme, kms,kme, &
                                       its,ite, jts,jte, kts,kte

   INTEGER, INTENT(IN )      ::        nlayers         ! total number of layers                                       

   LOGICAL, INTENT(IN )      ::        warm_rain

   INTEGER, INTENT(IN )      ::        ICLOUD
   INTEGER, INTENT(IN )      ::        MP_PHYSICS
!
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         INTENT(IN   ) ::                                    t3d, &
                                                             t8w, &
                                                             p8w, &
                                                             p3d, &
                                                             pi

   REAL, DIMENSION( ims:ime, jms:jme )                          , &
         INTENT(IN   )  ::                                 EMISS, &
                                                             TSK, &
                                                             ALBEDO, &
                                                             COSZEN                                                             

   REAL, INTENT(IN  )   ::                                   R,G

   REAL, DIMENSION( ims:ime, jms:jme )                          , &
         INTENT(IN   )  ::                                 XLAND, &
                                                            XICE, &
                                                            SNOW
!ccc Added for time-varying trace gases.
   INTEGER, INTENT(IN    ) ::                                 yr
   REAL, INTENT(IN    ) ::                                julian
!ccc

!
! Optional
!
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) ::                                         &
                                                        CLDFRA3D, &
                                                            QV3D, &
                                                            QC3D, &
                                                            QR3D, &
                                                            QI3D, &
                                                            QS3D, &
                                                            QG3D
   INTEGER, INTENT(IN):: has_reqc, has_reqi, has_reqs

   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) ::                                         &
                                                       F_ICE_PHY, &
                                                      F_RAIN_PHY

   LOGICAL, OPTIONAL, INTENT(IN)   ::                             &
                                   F_QV,F_QC,F_QR,F_QI,F_QS,F_QG

!  Ozone
   REAL, DIMENSION( ims:ime, kms:kme, jms:jme )                 , &
         OPTIONAL                                               , &
         INTENT(IN   ) :: O33D
   INTEGER, OPTIONAL, INTENT(IN ) :: o3input

      character(len=200) :: msg

!  LOCAL VARS
 
   REAL, DIMENSION( kts:kte+1 ) ::                          Pw1D, &
                                                            Tw1D

   REAL, DIMENSION( kts:kte ) ::                             CLDFRA1D, &
                                                             P1D, &
                                                             T1D, &
                                                            QV1D, &
                                                            QC1D, &
                                                            QR1D, &
                                                            QI1D, &
                                                           RHO1D, &
                                                            QS1D, &
                                                            QG1D, &
                                                            O31D


! Added local arrays for RRTMG
    integer ::                                              ncol, &
                                                            nlay, &
                                                         inflglw, &
                                                        iceflglw, &
                                                        liqflglw
! Dimension with extra layer from model top to TOA
    real, dimension( 1, kts:nlayers+1 )  ::                 plev, &
                                                            tlev
    real, dimension( 1, kts:nlayers )  ::                   play, &
                                                            tlay, &
                                                          h2ovmr, &
                                                           o3vmr, &
                                                          co2vmr, &
                                                           o2vmr, &
                                                          ch4vmr, &
                                                          n2ovmr, &
                                                        cfc11vmr, &
                                                        cfc12vmr, &
                                                        cfc22vmr, &
                                                         ccl4vmr
    real, dimension( kts:nlayers )  ::                     o3mmr

    real, dimension ( 1 ) ::                                tsfc, &
                                                              ps
    real ::                                                   ro, &
                                                            scon
!by xiaohui for training nn
    real, INTENT(INOUT  ), dimension( kms:kme-1, its:ite, jts:jte ) :: tlay_3d, &
                                                                     play_3d, &
                                                                     cldfrac_3d, &
                                                                     o3vmr_3d, &
                                                                     qv_3d, &
                                                                     qc_3d, &
                                                                     qr_3d, &
                                                                     qi_3d, &
                                                                     qs_3d, &
                                                                     qg_3d              

    real, INTENT(INOUT  ), dimension( kms:kme, its:ite, jts:jte ) :: tlev_3d, &
                                                                   plev_3d, &
                                                                   pi_3d

    real, INTENT(INOUT  ), dimension( its:ite, jts:jte ) :: landfrac_2d, &
                                                          icefrac_2d, &
                                                          snow_2d, &
                                                          tsfc_2d, &
                                                          emis_2d, &
                                                          solcon_2d, &
                                                          albedo_2d, &
                                                          coszen_2d

   REAL, INTENT(IN    )      ::                             SOLCON

! amontornes-bcodina 2015/09 solar eclipses
!  obscur --> degree of obscuration for solar eclipses prediction (2D)
                               REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: obscur

     
!..We can use message interface regardless of what options are running,
!.. so let us ask for it here.
      CHARACTER(LEN=256)                           :: message
      LOGICAL, EXTERNAL                            :: wrf_dm_on_monitor

!ccc To add time-varying trace gases (CO2, N2O and CH4). Read the conc.  from file
! then interpolate to date of run.
#ifdef CLWRFGHG
! CLWRF-UC June.09
      REAL(8)                                      :: co2, n2o, ch4, cfc11, cfc12
#else

! Set trace gas volume mixing ratios, 2005 values, IPCC (2007)
! carbon dioxide (379 ppmv) - this is being replaced by an annual function in v4.2
    real :: co2
!   data co2 / 379.e-6 / 
! methane (1774 ppbv)
    real :: ch4
    data ch4 / 1774.e-9 / 
! nitrous oxide (319 ppbv)
    real :: n2o
    data n2o / 319.e-9 / 
! cfc-11 (251 ppt)
    real :: cfc11
    data cfc11 / 0.251e-9 / 
! cfc-12 (538 ppt)
    real :: cfc12
    data cfc12 / 0.538e-9 / 
#endif
! cfc-22 (169 ppt)
    real :: cfc22
    data cfc22 / 0.169e-9 / 
! ccl4 (93 ppt)
    real :: ccl4
    data ccl4 / 0.093e-9 / 
! Set oxygen volume mixing ratio (for o2mmr=0.23143)
    real :: o2
    data o2 / 0.209488 /

    integer :: nb

! For old cloud property specification for rrtm_lw
! Cloud and precipitation absorption coefficients
    real :: abcw,abice,abrn,absn
    data abcw /0.144/
    data abice /0.0735/
    data abrn /0.330e-3/
    data absn /2.34e-3/

! Molecular weights and ratios for converting mmr to vmr units
!    real :: amd       ! Effective molecular weight of dry air (g/mol)  
!    real :: amw       ! Molecular weight of water vapor (g/mol)        
!    real :: amo       ! Molecular weight of ozone (g/mol)              
!    real :: amo2      ! Molecular weight of oxygen (g/mol)              
! Atomic weights for conversion from mass to volume mixing ratios                
!    data amd   /  28.9660   /                                                  
!    data amw   /  18.0160   /                                                  
!    data amo   /  47.9998   /                                                  
!    data amo2  /  31.9999   /
                                                                                 
    real :: amdw     ! Molecular weight of dry air / water vapor  
    real :: amdo     ! Molecular weight of dry air / ozone
    real :: amdo2    ! Molecular weight of dry air / oxygen
    data amdw /  1.607793 /                                                    
    data amdo /  0.603461 /
    data amdo2 / 0.905190 /
    
    real, dimension (1) :: landfrac, landm, snowh, icefrac

    integer :: pcols, pver

!
    INTEGER :: i,j,K, idx_rei
    REAL :: corr
    LOGICAL :: predicate

! Added for top of model adjustment.  Steven Cavallo NCAR/MMM December 2010
    INTEGER, PARAMETER :: nproflevs = 60 ! Constant, from the table
    INTEGER :: L, LL, klev               ! Loop indices      
    REAL, DIMENSION( kts:nlayers+1 ) :: varint
    REAL :: wght,vark,vark1,tem1,tem2,tem3
    REAL :: PPROF(nproflevs), TPROF(nproflevs)            
    ! Weighted mean pressure and temperature profiles from midlatitude 
    ! summer (MLS),midlatitude winter (MLW), sub-Arctic 
    ! winter (SAW),sub-Arctic summer (SAS), and tropical (TROP) 
    ! standard atmospheres.
    DATA PPROF   /1000.00,855.47,731.82,626.05,535.57,458.16,     &
                  391.94,335.29,286.83,245.38,209.91,179.57,      &
                  153.62,131.41,112.42,96.17,82.27,70.38,         &
                  60.21,51.51,44.06,37.69,32.25,27.59,            &
                  23.60,20.19,17.27,14.77,12.64,10.81,            &
                  9.25,7.91,6.77,5.79,4.95,4.24,                  &
                  3.63,3.10,2.65,2.27,1.94,1.66,                  &
                  1.42,1.22,1.04,0.89,0.76,0.65,                  &
                  0.56,0.48,0.41,0.35,0.30,0.26,                  &
                  0.22,0.19,0.16,0.14,0.12,0.10/
    DATA TPROF   /286.96,281.07,275.16,268.11,260.56,253.02,      &
                  245.62,238.41,231.57,225.91,221.72,217.79,      &
                  215.06,212.74,210.25,210.16,210.69,212.14,      &
                  213.74,215.37,216.82,217.94,219.03,220.18,      &
                  221.37,222.64,224.16,225.88,227.63,229.51,      &
                  231.50,233.73,236.18,238.78,241.60,244.44,      &
                  247.35,250.33,253.32,256.30,259.22,262.12,      &
                  264.80,266.50,267.59,268.44,268.69,267.76,      &
                  266.13,263.96,261.54,258.93,256.15,253.23,      &
                  249.89,246.67,243.48,240.25,236.66,233.86/    
!------------------------------------------------------------------
#if ( WRF_CHEM == 1 )
      IF ( aer_ra_feedback == 1) then
      IF ( .NOT. &
      ( PRESENT(tauaerlw1) .AND. &
        PRESENT(tauaerlw2) .AND. &
        PRESENT(tauaerlw3) .AND. &
        PRESENT(tauaerlw4) .AND. &
        PRESENT(tauaerlw5) .AND. &
        PRESENT(tauaerlw6) .AND. &
        PRESENT(tauaerlw7) .AND. &
        PRESENT(tauaerlw8) .AND. &
        PRESENT(tauaerlw9) .AND. &
        PRESENT(tauaerlw10) .AND. &
        PRESENT(tauaerlw11) .AND. &
        PRESENT(tauaerlw12) .AND. &
        PRESENT(tauaerlw13) .AND. &
        PRESENT(tauaerlw14) .AND. &
        PRESENT(tauaerlw15) .AND. &
        PRESENT(tauaerlw16) ) ) THEN
      CALL wrf_error_fatal  &
      ('Warning: missing fields required for aerosol radiation' )
      ENDIF
      ENDIF
#endif

!-----CALCULATE LONG WAVE RADIATION
!                                                              
! All fields are ordered vertically from bottom to top
! Pressures are in mb
!
! Annual function for co2 in WRF v4.2
      co2 = (280. + 90.*exp(0.02*(yr-2000)))*1.e-6

!ccc Read time-varying trace gases concentrations and interpolate them to run date.
!
#ifdef CLWRFGHG

   CALL read_CAMgases(yr,julian,"RRTMG",co2,n2o,ch4,cfc11,cfc12)

   IF ( wrf_dm_on_monitor() ) THEN
     WRITE(message,*)'CAM-CLWRF interpolated values______ year:',yr,' julian day:',julian
     call wrf_debug( 100, message)
     WRITE(message,*)'  CAM-CLWRF co2vmr: ',co2,' n2ovmr:',n2o,' ch4vmr:',ch4,' cfc11vmr:',cfc11,' cfc12vmr:',cfc12
     call wrf_debug( 100, message)
   ENDIF

#endif
!ccc

! latitude loop
  j_loop: do j = jts,jte

! longitude loop
     i_loop: do i = its,ite

         do k=kts,kte+1
            Pw1D(K) = p8w(I,K,J)/100.
            Tw1D(K) = t8w(I,K,J)
         enddo

         DO K=kts,kte
            QV1D(K)=0.
            QC1D(K)=0.
            QR1D(K)=0.
            QI1D(K)=0.
            QS1D(K)=0.
            CLDFRA1D(k)=0.
         ENDDO

         DO K=kts,kte
            QV1D(K)=QV3D(I,K,J)
            QV1D(K)=max(0.,QV1D(K))
         ENDDO

         IF (PRESENT(O33D)) THEN
            DO K=kts,kte
               O31D(K)=O33D(I,K,J)
            ENDDO
         ELSE
            DO K=kts,kte
               O31D(K)=0.0
            ENDDO
         ENDIF

         DO K=kts,kte
            T1D(K)=T3D(I,K,J)
            P1D(K)=P3D(I,K,J)/100.
         ENDDO

! moist variables

         IF (ICLOUD .ne. 0) THEN
            IF ( PRESENT( CLDFRA3D ) ) THEN
              DO K=kts,kte
                 CLDFRA1D(k)=CLDFRA3D(I,K,J)
              ENDDO
            ENDIF

            IF (PRESENT(F_QC) .AND. PRESENT(QC3D)) THEN
              IF ( F_QC) THEN
                 DO K=kts,kte
                    QC1D(K)=QC3D(I,K,J)
                    QC1D(K)=max(0.,QC1D(K))
                 ENDDO
              ENDIF
            ENDIF

            IF (PRESENT(F_QR) .AND. PRESENT(QR3D)) THEN
              IF ( F_QR) THEN
                 DO K=kts,kte
                    QR1D(K)=QR3D(I,K,J)
                    QR1D(K)=max(0.,QR1D(K))
                 ENDDO
              ENDIF
            ENDIF

! This logic is tortured because cannot test F_QI unless
! it is present, and order of evaluation of expressions
! is not specified in Fortran

            IF ( PRESENT ( F_QI ) ) THEN
              predicate = F_QI
            ELSE
              predicate = .FALSE.
            ENDIF

! For MP option 3
            IF (.NOT. predicate .and. .not. warm_rain) THEN
               DO K=kts,kte
                  IF (T1D(K) .lt. 273.15) THEN
                  QI1D(K)=QC1D(K)
                  QS1D(K)=QR1D(K)
                  QC1D(K)=0.
                  QR1D(K)=0.
                  ENDIF
               ENDDO
            ENDIF

            IF (PRESENT(F_QI) .AND. PRESENT(QI3D)) THEN
               IF (F_QI) THEN
                  DO K=kts,kte
                     QI1D(K)=QI3D(I,K,J)
                     QI1D(K)=max(0.,QI1D(K))
                  ENDDO
               ENDIF
            ENDIF

            IF (PRESENT(F_QS) .AND. PRESENT(QS3D)) THEN
               IF (F_QS) THEN
                  DO K=kts,kte
                     QS1D(K)=QS3D(I,K,J)
                     QS1D(K)=max(0.,QS1D(K))
                  ENDDO
               ENDIF
            ENDIF

            IF (PRESENT(F_QG) .AND. PRESENT(QG3D)) THEN
               IF (F_QG) THEN
                  DO K=kts,kte
                     QG1D(K)=QG3D(I,K,J)
                     QG1D(K)=max(0.,QG1D(K))
                  ENDDO
               ENDIF
            ENDIF

! mji - For MP option 5
            IF ( PRESENT(F_QI) .and. PRESENT(F_QC) .and. PRESENT(F_QS) .and. PRESENT(F_ICE_PHY) ) THEN
               IF ( F_QC .and. .not. F_QI .and. F_QS ) THEN
                  DO K=kts,kte
                     qi1d(k) = 0.1*qs3d(i,k,j)
                     qs1d(k) = 0.9*qs3d(i,k,j)
                     qc1d(k) = qc3d(i,k,j)
                     qi1d(k) = max(0.,qi1d(k))
                     qc1d(k) = max(0.,qc1d(k))
                  ENDDO
               ENDIF
            ENDIF

        ENDIF

!   For mp option=5 or 85  (new Ferrier- Aligo or fer_hires scheme), QI3D saves all
#if (HWRF == 1)
        IF ( mp_physics == FER_MP_HIRES .OR. &
             mp_physics == FER_MP_HIRES_ADVECT .OR. &
             mp_physics == ETAMP_HWRF ) THEN
#else
        IF ( mp_physics == FER_MP_HIRES .OR. &
             mp_physics == FER_MP_HIRES_ADVECT) THEN
#endif
                  DO K=kts,kte
                     qi1d(k) = qi3d(i,k,j)
                     qs1d(k) = 0.0
                     qc1d(k) = qc3d(i,k,j)
                     qi1d(k) = max(0.,qi1d(k))
                     qc1d(k) = max(0.,qc1d(k))
                  ENDDO
        ENDIF

         DO K=kts,kte
            QV1D(K)=AMAX1(QV1D(K),1.E-12) 
         ENDDO

! Set up input for longwave
         ncol = 1
! Add extra layer from top of model to top of atmosphere
!         nlay = (kte - kts + 1) + 1
! Edited for top of model adjustment (nlayers = kte + 1).  
! Steven Cavallo, December 2010
          nlay = nlayers ! Keep these indices the same

! Select cloud liquid and ice optics parameterization options
! For passing in cloud optical properties directly:
!         inflglw = 0
!         iceflglw = 0
!         liqflglw = 0
! For passing in cloud physical properties; cloud optics parameterized in RRTMG:
         inflglw = 2
         iceflglw = 3
         liqflglw = 1

!Mukul change the flags here with reference to the new effective cloud/ice/snow radius
         IF (ICLOUD .ne. 0) THEN

! special case for P3 microphysics
! put ice into snow category for optics, then set ice to zero
            IF (has_reqs .eq. 0 .and. has_reqi .ne. 0 .and. has_reqc .ne. 0) THEN
               inflglw  = 5
               iceflglw = 5
               DO K=kts,kte
                  QS1D(K)=QI3D(I,K,J)
                  QI1D(K)=0.
               END DO
            END IF

         ENDIF

! Layer indexing goes bottom to top here for all fields.
! Water vapor and ozone are converted from mmr to vmr. 
! Pressures are in units of mb here. 
         plev(ncol,1) = pw1d(1)
         tlev(ncol,1) = tw1d(1)
         tsfc(ncol) = tsk(i,j)
         do k = kts, kte
            play(ncol,k) = p1d(k)
            plev(ncol,k+1) = pw1d(k+1)
            tlay(ncol,k) = t1d(k)
            tlev(ncol,k+1) = tw1d(k+1)

         enddo

!  Set up values for extra layers to the top of the atmosphere.                       
!  Temperature is calculated based on an average temperature profile given
!  here in a table.  The input table data is linearly interpolated to the
!  column pressure.  Mixing ratios are held constant except for ozone.  
!  Caution should be used if model top pressure is less than 5 hPa.
!  Steven Cavallo, NCAR/MMM, December 2010
       ! Calculate the column pressure buffer levels above the 
       ! model top       
       do L=kte+1,nlayers,1
          plev(ncol,L+1) = plev(ncol,L) - deltap
          play(ncol,L) = 0.5*(plev(ncol,L) + plev(ncol,L+1))
! Fill in height array above model top to top of atmosphere using
! dz from model top layer for completeness, though this information is not
! likely to be used by the exponential-random cloud overlap method.
       enddo          
       ! Add zero as top level.  This gets the temperature max at the
       ! stratopause, reducing the downward flux errors in the top 
       ! levels.  If zero happened to be the top level already,
       ! this will add another level with zero, but will not affect
       ! the radiative transfer calculation.
       plev(ncol,nlayers+1) = 0.00
       play(ncol,nlayers) =  0.5*(plev(ncol,nlayers) + plev(ncol,nlayers+1))

       ! Interpolate the table temperatures to column pressure levels    
       do L=1,nlayers+1,1
          if ( PPROF(nproflevs) .lt. plev(ncol,L) ) then
             do LL=2,nproflevs,1       
                if ( PPROF(LL) .lt. plev(ncol,L) ) then           
                   klev = LL - 1
                   exit
                endif
             enddo
          
          else
             klev = nproflevs
          endif  
  
          if (klev .ne. nproflevs ) then
             vark  = TPROF(klev) 
             vark1 = TPROF(klev+1)
             wght=(plev(ncol,L)-PPROF(klev) )/( PPROF(klev+1)-PPROF(klev))
          else
             vark  = TPROF(klev) 
             vark1 = TPROF(klev)
             wght = 0.0
          endif
          varint(L) = wght*(vark1-vark)+vark

       enddo                   
       
       ! Match the interpolated table temperature profile to WRF column                    
       do L=kte+1,nlayers+1,1
          tlev(ncol,L) = varint(L) + (tlev(ncol,kte) - varint(kte))
          !if ( L .le. nlay ) then
          tlay(ncol,L-1) = 0.5*(tlev(ncol,L) + tlev(ncol,L-1))  
          !endif
       enddo   

! End top of model buffer 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
! Get ozone profile including amount in extra layer above model top.
! Steven Cavallo: Must pass nlay-1 into subroutine to get nlayers 
! dimension for o3mmr
         call inirad (o3mmr,plev,kts,kte)

! Steven Cavallo: Changed to nlayers from kte+1
        if(present(o33d)) then
         do k = kts, nlayers
            o3vmr(ncol,k) = o3mmr(k) * amdo
            IF ( PRESENT( O33D ) ) THEN
            if(o3input .eq. 2)then
               if(k.le.kte)then
                 o3vmr(ncol,k) = o31d(k)
               else
! apply shifted climatology profile above model top
                 o3vmr(ncol,k) = o31d(kte) - o3mmr(kte)*amdo + o3mmr(k)*amdo
                 if(o3vmr(ncol,k) .le. 0.)o3vmr(ncol,k) = o3mmr(k)*amdo
               endif
            endif
            ENDIF
         enddo
        else
         do k = kts, nlayers
            o3vmr(ncol,k) = o3mmr(k) * amdo            
         enddo
        endif

! Compute cloud water/ice paths and particle sizes for input to radiation (CAM method)
            pcols = ncol
            landfrac(ncol) = 2.-XLAND(I,J)
            landm(ncol) = landfrac(ncol)
            icefrac(ncol) = XICE(I,J)

!by xiaohui
         landfrac_2d(i,j)=landfrac(ncol)
         icefrac_2d(i,j)=icefrac(ncol)
         snow_2d(i,j)=snow(i,j)
         tsfc_2d(i,j) = tsfc(ncol)
         emis_2d(i,j)=emiss(i,j)
         albedo_2d(i,j)=albedo(i,j)
         coszen_2d(i,j)=coszen(i,j)

         plev_3d(1,i,j) = plev(1,1)
         tlev_3d(1,i,j) = tlev(1,1)
         pi_3d(1,i,j) = pi(i,1,j)         
         do k = kts, kte
            play_3d(k,i,j)=play(1,k)
            plev_3d(k+1,i,j)= plev(1,k+1)
            pi_3d(k+1,i,j) = pi(i,k+1,j)         
            tlay_3d(k,i,j) = tlay(1,k)
            tlev_3d(k+1,i,j)= tlev(1,k+1)
            qv_3d(k,i,j) = qv1d(k)
            qc_3d(k,i,j) = qc1d(k)
            qr_3d(k,i,j) = qr1d(k)
            qi_3d(k,i,j) = qi1d(k)
            qs_3d(k,i,j) = qs1d(k)
            qg_3d(k,i,j) = qg1d(k)

            cldfrac_3d(k,i,j) = cldfra1d(k)

            o3vmr_3d(k,i,j)=o3vmr(ncol,k)

         enddo

! amontornes-bcodina 2015/09 solar eclipses
         scon = solcon*(1-obscur(i,j))
         solcon_2d(i,j)=scon          

      end do i_loop
   end do j_loop                                           

!-------------------------------------------------------------------

   END SUBROUTINE rrtmg_lw_sw_preprocess

END MODULE module_ra_rrtmg_lw_sw
