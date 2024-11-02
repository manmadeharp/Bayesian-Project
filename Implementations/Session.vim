let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/source
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 Bayesian-Project
badd +1 Bayesian-Project/Random_Walk_Metropolis_Hastings.ipynb
badd +2 ~/PycharmProjects/Bayesian-Project/rwh.py
badd +78 ~/AppData/Local/nvim/init.lua
badd +28 term://~/source//17460:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +1 term://~/source//23388:.
badd +193 ~/PycharmProjects/Bayesian-Project/rwh_univariable.py
badd +1 term://~/source//20792:.
badd +3761 term://~/source//16940:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +585 term://~/source//16884:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +374 term://~/source//6708:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +1 ./Implementations/RandomWalkMetropolisHastings.py
badd +10 ~/PycharmProjects/Bayesian-Project/Implementations/RandomWalkMetropolisHastings.py
badd +3949 term://~/source//6576:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +5 ~/PycharmProjects/Bayesian-Project/Implementations/MetropolisHastings/RandomWalkMetropolisHastings.py
badd +44 ~/PycharmProjects/Bayesian-Project/Implementations/MetropolisHastings/Distributions.py
badd +43 term://~/PycharmProjects/Bayesian-Project/Implementations//4708:C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
badd +1 ~/AppData/Local/nvim/lua/plugins.lua
argglobal
%argdel
$argadd Bayesian-Project
edit ~/AppData/Local/nvim/lua/plugins.lua
argglobal
balt ~/AppData/Local/nvim/init.lua
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 1 - ((0 * winheight(0) + 34) / 69)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
lcd ~/PycharmProjects/Bayesian-Project/Implementations
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
