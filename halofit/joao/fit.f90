subroutine fit(n_features, iter, l1_reg, l2_reg, W, Wi, Wj, X, Xi, Xj, R, Ri, Rj, normcolx)
    integer, intent(in) :: n_features, Wi, Wj, Xi, Xj, Ri, Rj
    real(8), intent(in) :: l1_reg, l2_reg
    real(8), dimension(Xi, Xj), intent(in)  :: X
    real(8), dimension(Xj), intent(in) :: normcolx
    real(8), dimension(Wi, Wj), intent(inout) :: W
    real(8), dimension(Ri, Rj), intent(inout) :: R
    real(8), dimension(Wi) :: wii
    real(8), dimension(Xi) :: Xii    
    real(8), dimension(Rj) :: tmp
    real(8) :: normtmp, reg, DNRM2
    integer :: n_iter, f_iter, iter
    
    do n_iter = 1, iter 
        do f_iter = 1, Xj 
            do j = 1, Wi
                wii(j) = W(j, f_iter)
            end do

            do j = 1, Xi
                 Xii(j) = X(j, f_iter)   
            end do 

            call DGER(Ri, Rj, 1.0d0, Xii, 1, wii, 1, R, Ri)    
            call DGEMV('T', Ri, Rj, 1.0d0, R, Ri, Xii, 1, 0.0d0, tmp, 1)

            ! print*, tmp
            normtmp = DNRM2(Rj, tmp, 1)
            ! print*, normtmp   

            if (1.d0  - l1_reg/normtmp >= 0.0d0) then
                reg =1.0d0  - l1_reg/normtmp 
            else
                reg = 0.0d0
            end if

            call DCOPY(Rj, tmp, 1, wii, 1)
            call DSCAL(Rj, reg/(normcolx(f_iter) + l2_reg), wii, 1)
            call DGER(Ri, Rj, -1.0d0, Xii, 1, wii, 1, R, Ri)

            do j = 1, Wi
                W(j, f_iter) = wii(j) 
                ! print*, W(j, 2)
            end do
        
        end do
    end do
end subroutine fit
