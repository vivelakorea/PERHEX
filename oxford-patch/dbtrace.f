!     Per-increment GB-distance update by ray-tracing the DEFORMED grain
!     boundary facets. Facet quads are tracked by their actual NODES:
!     the deck writes GB nodal displacements to the .fil each increment
!     (*NODE FILE, nset=GBN), URDFIL reads them, and each deformed
!     vertex is x = X0(node) + u(node) + F_macro.offset (the offset term
!     unwraps periodic images; the PBC *Equations guarantee it exactly).
!     Ray origins are the elements' current centroids (UMAT COORDS),
!     directions the current lattice-rotated slip directions stored by
!     solve(). The updated distances feed the NEXT increment (explicit
!     in geometry, consistent with the explicit state update).
!
!     Precomputed tables come from gen_gb_rt.py (gb_rt.bin).
      module dbtrace
      implicit none
!     tables (gb_rt.bin)
      integer :: rt_nel=0, rt_nfac=0, rt_nsys=0, rt_ncand=0, rt_nnod=0
      real(8) :: rt_lphys = 1.d0
      real(8), allocatable :: rt_o(:,:)       ! (3,nel) unwrap offsets
      real(8), allocatable :: rt_v0(:)        ! (nel) ref volumes
      real(8), allocatable :: rt_x0(:,:)      ! (3,nnode) ref nodal coords
      integer, allocatable :: rt_fnode(:,:)   ! (4,nfac) vertex node ids
      real(8), allocatable :: rt_foff(:,:,:)  ! (3,4,nfac) vertex offsets
      integer, allocatable :: rt_ptr(:)       ! (nel*nsys+1) CSR
      integer, allocatable :: rt_idx(:)       ! (ncand)
!     per-increment state written by the material routines / URDFIL
      real(8), allocatable :: rt_coords(:,:)  ! (3,nel) current centroid
      real(8), allocatable :: rt_f(:,:,:)     ! (3,3,nel) current F
      real(8), allocatable :: rt_dirs(:,:,:)  ! (3,nsys,nel) current slip dir
      real(8), allocatable :: rt_u(:,:)       ! (3,nnode) nodal displacement
!     output: current distances (um), used by solve when dbconvect==2
      real(8), allocatable :: rt_db(:,:)      ! (nsys,nel)
!     GB node user ids (gbnodes.txt) for .fil internal-label mapping
      integer :: rt_ngb = 0
      integer, allocatable :: rt_gbn(:)
      logical :: rt_loaded = .false.
      integer :: rt_ncall = 0
      character(len=256) :: rt_dir = ''
      integer :: rt_lendir = 0
      contains
!
      subroutine dbtrace_init(numel)
      use globalvariables, only: db_all
      integer, intent(in) :: numel
      character(len=256) :: dir
      integer :: lendir, i, k
      logical :: ex
      integer :: hdr(5)
      call getoutdir(dir, lendir)
      rt_dir = dir
      rt_lendir = lendir
      inquire(file=dir(:lendir)//'/gb_rt.bin', exist=ex)
      if (.not. ex) return
      open(398, file=dir(:lendir)//'/gb_rt.bin', access='stream',
     + form='unformatted', action='read', status='old')
      read(398) hdr
      rt_nel = hdr(1); rt_nfac = hdr(2); rt_nsys = hdr(3)
      rt_ncand = hdr(4); rt_nnod = hdr(5)
      read(398) rt_lphys
      if (rt_nel /= numel) then
          write(*,*) 'gb_rt.bin element count mismatch!', rt_nel, numel
          call xit
      end if
      allocate(rt_o(3,rt_nel), rt_v0(rt_nel), rt_x0(3,rt_nnod))
      allocate(rt_fnode(4,rt_nfac), rt_foff(3,4,rt_nfac))
      allocate(rt_ptr(rt_nel*rt_nsys+1), rt_idx(rt_ncand))
!     python writes C-order; stream fills Fortran arrays col-major with
!     the LAST python axis fastest, hence the reversed shapes above
      read(398) rt_o
      read(398) rt_v0
      read(398) rt_x0
      read(398) rt_fnode
      read(398) rt_foff
      read(398) rt_ptr
      read(398) rt_idx
      close(398)
      allocate(rt_coords(3,rt_nel), rt_f(3,3,rt_nel))
      allocate(rt_dirs(3,rt_nsys,rt_nel), rt_db(rt_nsys,rt_nel))
      allocate(rt_u(3,rt_nnod))
      rt_coords = 0.d0; rt_f = 0.d0; rt_dirs = 0.d0; rt_u = 0.d0
!     start from the exact t=0 table (um)
      do i = 1, rt_nel
          do k = 1, rt_nsys
              rt_db(k,i) = db_all(i,k)
          end do
      end do
!     GB node id list (user numbering, ascending)
      inquire(file=dir(:lendir)//'/gbnodes.txt', exist=ex)
      if (.not. ex) then
          write(*,*) 'gbnodes.txt missing next to gb_rt.bin!'
          call xit
      end if
      open(399, file=dir(:lendir)//'/gbnodes.txt', action='read',
     + status='old')
      rt_ngb = 0
      do
          read(399,*,end=10) i
          rt_ngb = rt_ngb + 1
      end do
   10 rewind(399)
      allocate(rt_gbn(rt_ngb))
      do i = 1, rt_ngb
          read(399,*) rt_gbn(i)
      end do
      close(399)
      rt_loaded = .true.
      write(*,*) 'gb_rt.bin read: ', rt_nfac, ' facets, ',
     + rt_nnod, ' nodes, ', rt_ncand, ' candidates, ',
     + rt_ngb, ' GB nodes'
      end subroutine dbtrace_init
!
      subroutine dbtrace_update
!     rebuild deformed facet vertices from nodal positions and
!     ray-trace all (element, system, +-sense) candidate lists
      use globalvariables, only: db_all
      real(8) :: fmac(3,3), vsum, vd(3,4), orig(3), dvec(3)
      real(8) :: e1(3), e2(3), pv(3), sv(3), qv(3)
      real(8) :: det, u, w, t, tbest, dnew
      real(8) :: rsum, rmin, r
      real(8), parameter :: rtcap = 0.05d0
      integer :: i, k, s, j, jf, n, sen, nmiss, cnt
      if (.not. rt_loaded) return
!     macroscopic F (volume-weighted average; equals the PBC macro F)
      fmac = 0.d0; vsum = 0.d0
      do i = 1, rt_nel
          fmac = fmac + rt_v0(i)*rt_f(:,:,i)
          vsum = vsum + rt_v0(i)
      end do
      fmac = fmac/vsum
      nmiss = 0
!$omp parallel do private(i,s,k,j,jf,n,sen,vd,orig,dvec,e1,e2,pv,sv,
!$omp+ qv,det,u,w,t,tbest,dnew) reduction(+:nmiss) schedule(dynamic,64)
      do i = 1, rt_nel
          do s = 1, rt_nsys
              dnew = 0.d0
              do sen = 0, 1
                  tbest = huge(1.d0)
                  orig = rt_coords(:,i) + matmul(fmac, rt_o(:,i))
                  dvec = rt_dirs(:,s,i)
                  if (sen == 1) dvec = -dvec
!                 CSR bounds: python offsets are 0-based, rt_idx 1-based
                  do j = rt_ptr((i-1)*rt_nsys+s)+1,
     +                   rt_ptr((i-1)*rt_nsys+s+1)
                      jf = rt_idx(j)
!                     deformed facet vertices from tracked nodes
                      do k = 1, 4
                          n = rt_fnode(k,jf)
                          vd(:,k) = rt_x0(:,n) + rt_u(:,n)
     +                        + matmul(fmac, rt_foff(:,k,jf))
                      end do
!                     two triangles: (1,2,3) and (1,3,4)
                      do k = 0, 1
                          e1 = vd(:,2+k) - vd(:,1)
                          e2 = vd(:,3+k) - vd(:,1)
                          pv(1) = dvec(2)*e2(3) - dvec(3)*e2(2)
                          pv(2) = dvec(3)*e2(1) - dvec(1)*e2(3)
                          pv(3) = dvec(1)*e2(2) - dvec(2)*e2(1)
                          det = pv(1)*e1(1)+pv(2)*e1(2)+pv(3)*e1(3)
                          if (dabs(det) < 1.d-14) cycle
                          sv = orig - vd(:,1)
                          u = (pv(1)*sv(1)+pv(2)*sv(2)+pv(3)*sv(3))/det
                          if (u < -1.d-6 .or. u > 1.d0+1.d-6) cycle
                          qv(1) = sv(2)*e1(3) - sv(3)*e1(2)
                          qv(2) = sv(3)*e1(1) - sv(1)*e1(3)
                          qv(3) = sv(1)*e1(2) - sv(2)*e1(1)
                          w = (qv(1)*dvec(1)+qv(2)*dvec(2)
     +                        +qv(3)*dvec(3))/det
                          if (w < -1.d-6 .or. u+w > 1.d0+1.d-6) cycle
                          t = (qv(1)*e2(1)+qv(2)*e2(2)
     +                        +qv(3)*e2(3))/det
                          if (t > 1.d-9 .and. t < tbest) tbest = t
                      end do
                  end do
                  if (tbest < huge(1.d0)) then
                      if (dnew == 0.d0 .or. tbest*rt_lphys < dnew)
     +                    dnew = tbest*rt_lphys
                  end if
              end do
              if (dnew > 0.d0) then
!                 rate-limit the update: the raw facet-tracked distance
!                 steps discontinuously when the hit facet changes, and
!                 feeding the step into the explicit hardening update
!                 collapses dt (measured: ms-scale increments). The
!                 capped value follows the geometry within a few
!                 increments while keeping dtauc bounded.
                  rt_db(s,i) = min(max(dnew, rt_db(s,i)*(1.d0-rtcap)),
     +                             rt_db(s,i)*(1.d0+rtcap))
              else
                  nmiss = nmiss + 1   ! keep previous value
              end if
          end do
      end do
!$omp end parallel do
      if (nmiss > 0) write(*,*) 'dbtrace: ', nmiss, ' rays missed'
!
!     drift log (per update) + latest-state dump for post-processing
      rt_ncall = rt_ncall + 1
      rsum = 0.d0; rmin = huge(1.d0); cnt = 0
      do i = 1, rt_nel
          do s = 1, rt_nsys
              if (db_all(i,s) < 1.d19) then
                  r = rt_db(s,i)/db_all(i,s)
                  rsum = rsum + r
                  if (r < rmin) rmin = r
                  cnt = cnt + 1
              end if
          end do
      end do
      open(396, file=rt_dir(:rt_lendir)//'/dbdrift.log',
     +     position='append', action='write')
      write(396,'(i8,2es15.6)') rt_ncall, rsum/max(cnt,1), rmin
      close(396)
      open(395, file=rt_dir(:rt_lendir)//'/db_rt_last.bin',
     +     access='stream', form='unformatted', status='replace',
     +     action='write')
      write(395) rt_db
      close(395)
      end subroutine dbtrace_update
!
      end module dbtrace
!
!
!     Read GB nodal displacements from the results (.fil) file at the
!     end of every increment, then refresh the GB distances. Record
!     key 101 = nodal U (label, u1, u2, u3). LOVRWRT=1 keeps the .fil
!     small (each increment overwrites the last).
      SUBROUTINE URDFIL(LSTOP,LOVRWRT,KSTEP,KINC,DTIME,TIME)
!     .fil labels are INTERNAL numbers when the model is assembly-form
!     (instances shift them; here dummies take 1-3). The nset is written
!     in ascending order on both sides, so the constant shift is
!     detected by aligning the k-th record with the k-th gbnodes entry.
      use dbtrace, only: rt_loaded, rt_nnod, rt_u, rt_ngb, rt_gbn,
     + dbtrace_update
      use userinputs, only: dbconvect
      INCLUDE 'ABA_PARAM.INC'
      DIMENSION ARRAY(513), JRRAY(NPRECD,513), TIME(2)
      EQUIVALENCE (ARRAY(1),JRRAY(1,1))
      INTEGER :: JRCD, KEY, NODE, NBAD, KREC, IOFF
      IF (.NOT. rt_loaded) RETURN
      CALL POSFIL(KSTEP,KINC,ARRAY,JRCD)
      NBAD = 0
      KREC = 0
      IOFF = 0
      DO WHILE (.TRUE.)
          CALL DBFILE(0,ARRAY,JRCD)
          IF (JRCD .NE. 0) EXIT
          KEY = JRRAY(1,2)
          IF (KEY .EQ. 101) THEN
              KREC = KREC + 1
              NODE = JRRAY(1,3)
              IF (KREC .EQ. 1) IOFF = NODE - rt_gbn(1)
              NODE = NODE - IOFF
              IF (KREC .LE. rt_ngb) THEN
                  IF (NODE .NE. rt_gbn(KREC)) NBAD = NBAD + 1
              END IF
              IF (NODE .GE. 1 .AND. NODE .LE. rt_nnod) THEN
                  rt_u(1,NODE) = ARRAY(4)
                  rt_u(2,NODE) = ARRAY(5)
                  rt_u(3,NODE) = ARRAY(6)
              END IF
          END IF
      END DO
      IF (NBAD .GT. 0 .OR. KREC .NE. rt_ngb) THEN
          WRITE(*,*) 'URDFIL: node map mismatch! records=', KREC,
     +     ' expected=', rt_ngb, ' misaligned=', NBAD
          LSTOP = 1
      END IF
      LOVRWRT = 1
      IF (dbconvect == 2) CALL dbtrace_update
      RETURN
      END SUBROUTINE URDFIL
