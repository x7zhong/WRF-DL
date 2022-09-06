!------------------------------------------------------------------
MODULE module_ra_rrtmg_lw_sw_dl_replace 

  use, intrinsic :: iso_c_binding
  implicit none

! CHARACTER(*) :: wrf_model_infer_err_message
!  INTEGER,PARAMETER :: model_input_shape_c = 1036
!  INTEGER,PARAMETER :: model_output_shape_c = 228
  !REAL(c_float), DIMENSION(1024*1036):: model_inputdata
  !REAL(c_float), DIMENSION(1024*228 ):: model_outputdata

    save

    interface

!--------------------------------------subroutine--------------------------

    subroutine infer_init(use_gpu) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none
          integer(c_int), value :: use_gpu 
    end subroutine infer_init

    subroutine infer_run(                                                                         &
          emis, shape_x, shape_y, shape_z, shape_z_lev,                                           &
          solcon,                                                                                 &
          albedo,                                                                                 &
          landfrac,                                                                               &   
          icefrac,                                                                                &
          snow,                                                                                   &   
          coszen,                                                                                 &              
          tsfc,                                                                                   &
          tlay,                                                                                   &
          tlev,                                                                                   &
          play,                                                                                   &
          plev,                                                                                   &
          qv,                                                                                     &
          qc,                                                                                     &
          qr,                                                                                     &
          qi,                                                                                     &
          qs,                                                                                     &
          qg,                                                                                     &
          o3vmr,                                                                                  &
          cldfrac,                                                                                &              
          pi,                                                                                     &
          rthraten,                                                                               &
          rthratenlw,                                                                             &
          rthratensw,                                                                             &
          lwuflx,                                                                                 &
          lwdflx,                                                                                 &
          swuflx,                                                                                 &
          swdflx) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none

          integer(c_int), value :: shape_x
          integer(c_int), value :: shape_y
          integer(c_int), value :: shape_z
          integer(c_int), value :: shape_z_lev
          REAL(c_float), DIMENSION( shape_x, shape_y ), INTENT(IN   )  :: emis, &
                                                                          solcon, &
                                                                          albedo, &
                                                                          landfrac, &
                                                                          icefrac, &
                                                                          snow, &
                                                                          coszen, &
                                                                          tsfc          

          REAL(c_float), DIMENSION(shape_z_lev, shape_x, shape_y), INTENT(IN) :: tlev, &
                                                                                 plev, &
                                                                                 swuflx, &
                                                                                 swdflx, &
                                                                                 lwuflx, &
                                                                                 lwdflx, & 
                                                                                 rthraten, &
                                                                                 rthratenlw, &
                                                                                 rthratensw, &
                                                                                 pi          

          REAL(c_float), DIMENSION(shape_z, shape_x, shape_y), INTENT(IN) :: play, &
                                                                             tlay, &
                                                                             qv, &
                                                                             qc, &
                                                                             qr, &
                                                                             qi, &
                                                                             qs, &
                                                                             qg, &
                                                                             o3vmr, &
                                                                             cldfrac

    end subroutine infer_run

    subroutine save_fortran_array2(data_array, in_x, in_y, filename) bind (c)
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: in_x
        integer(c_int), value :: in_y
        REAL(c_float), DIMENSION( in_x, in_y ), INTENT(IN)  :: data_array
        CHARACTER(c_char), DIMENSION(*) :: filename
    end subroutine save_fortran_array2

    subroutine save_fortran_array3(data_array, in_x, in_y, in_z, filename) bind (c)
        use, intrinsic :: iso_c_binding
        implicit none
        integer(c_int), value :: in_x
        integer(c_int), value :: in_y
        integer(c_int), value :: in_z
        REAL(c_float), DIMENSION( in_x, in_y, in_z), INTENT(IN)  :: data_array
        CHARACTER(c_char), DIMENSION(*) :: filename
    end subroutine save_fortran_array3

    subroutine save_fortran_array(                                                                &
          ims, ime, jms, jme, kms, kme,                                                           &
          its, ite, jts, jte, kts, kte,                                                           &
          emis,                                                                                   &
          solcon,                                                                                 &
          albedo,                                                                                 &
          landfrac,                                                                               &   
          icefrac,                                                                                &
          snow,                                                                                   &   
          coszen,                                                                                 &              
          tsfc,                                                                                   &
          tlay,                                                                                   &
          tlev,                                                                                   &
          play,                                                                                   &
          plev,                                                                                   &
          qv,                                                                                     &
          qc,                                                                                     &
          qr,                                                                                     &
          qi,                                                                                     &
          qs,                                                                                     &
          qg,                                                                                     &
          o3vmr,                                                                                  &
          cldfrac,                                                                                &   
          swuflx,                                                                                 &   
          swdflx,                                                                                 &   
          lwuflx,                                                                                 &   
          lwdflx,                                                                                 &
          lwhr,                                                                                   &
          swhr,                                                                                   &
          rthraten,                                                                               &          
          rthratenlw,                                                                             &          
          rthratensw,                                                                             &          
          pi,                                                                                     &          
          glw,                                                                                    &          
          gsw,                                                                                    &          
          julday, gmt, xtime) bind (c)
          use, intrinsic :: iso_c_binding
          implicit none

          integer(c_int), value :: ims
          integer(c_int), value :: ime
          integer(c_int), value :: jms
          integer(c_int), value :: jme
          integer(c_int), value :: kms
          integer(c_int), value :: kme

          integer(c_int), value :: its
          integer(c_int), value :: ite
          integer(c_int), value :: jts
          integer(c_int), value :: jte
          integer(c_int), value :: kts
          integer(c_int), value :: kte

          integer(c_int), value :: julday
          REAL(c_float), value :: gmt
          REAL(c_float), value :: xtime
    
          REAL(c_float), DIMENSION( ims:ime, jms:jme ), INTENT(IN   )  :: emis, &
                                                                          solcon, &
                                                                          albedo, &
                                                                          landfrac, &
                                                                          icefrac, &
                                                                          snow, &
                                                                          coszen, &
                                                                          tsfc, &
                                                                          glw, &
                                                                          gsw

          REAL(c_float), DIMENSION(kms:kme-1, ims:ime, jms:jme), INTENT(IN) :: play, &
                                                                               tlay, &
                                                                               qv, &
                                                                               qc, &
                                                                               qr, &
                                                                               qi, &
                                                                               qs, &
                                                                               qg, &
                                                                               o3vmr, &
                                                                               cldfrac, &
                                                                               lwhr, &
                                                                               swhr

          REAL(c_float), DIMENSION(kms:kme, ims:ime, jms:jme), INTENT(IN) :: tlev, &
                                                                             plev, &
                                                                             swuflx, &
                                                                             swdflx, &
                                                                             lwuflx, &
                                                                             lwdflx, & 
                                                                             rthraten, &
                                                                             rthratenlw, &
                                                                             rthratensw, &
                                                                             pi

    end subroutine save_fortran_array

  end interface

  CONTAINS
    subroutine fillOneDimArray_atfortran (a)      
        implicit none      
        real,dimension (:), intent (out) :: a      

        ! local variables     
        integer :: i, arraySize  
        arraySize = size(a)
        do i = 1, arraySize         
            a(i) = 2.0      
        end do  

    end subroutine fillOneDimArray_atfortran 

    subroutine printOneDimArray_atfortran(a)
        implicit none

        real,dimension (:) :: a  
        integer::i, arraySize
        !arraySize = size(a)
        !print *, "size=", arraySize, ":", a
        arraySize = 10
        do i = 1, arraySize
            Print *, a(i)
            !WRITE(wrf_model_infer_err_message, *),a(i) 
            !call wrf_debug(0,wrf_model_infer_err_message)
        end do

    end subroutine printOneDimArray_atfortran

END MODULE module_ra_rrtmg_lw_sw_dl_replace
