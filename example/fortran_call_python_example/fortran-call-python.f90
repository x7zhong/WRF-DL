! ------------------------------------------------------
! Hello World from python
! ------------------------------------------------------
program  call_python
use, intrinsic :: iso_c_binding
  implicit none

  interface
  
!--------------------------------------subroutine--------------------------
  subroutine hello_world() bind (c)
  end subroutine hello_world  
!--------------------------------------subroutine--------------------------

  end interface

  call hello_world()

end program call_python                           
