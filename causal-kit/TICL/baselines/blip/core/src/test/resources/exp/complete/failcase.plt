set terminal png size 1200,1200 enhanced font "Helvetica,20"
set output 'failcase.png'

set multiplot layout 2,1 rowsfirst

set ylabel "k2"
set xlabel "k1"

plot "failcase-1-ko.res" t "BC > BD, I miss" w p ls 1, \
     "failcase-1-ok.res" t "BC > BD, I guess" w p ls 2, \
     "failcase-2-ko.res" t "BD > BC, I miss" w p ls 3, \
     "failcase-2-ok.res" t "BD > BC, I guess" w p ls 4;

plot  "failcase-2-ko.res" w p ls 3, \
      "failcase-1-ko.res" w p ls 1;

unset multiplot