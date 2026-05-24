set terminal png size 1200,1200 enhanced font "Helvetica,20"
set output 'failcase-sc.png'

set multiplot layout 3,2 rowsfirst

set ylabel "k2"
set xlabel "k1"

plot "failcase-sc-0.51.res" t "D > B" w p ls 1;
plot "failcase-sc-0.60.res" t "D > B" w p ls 1;
plot "failcase-sc-0.70.res" t "D > B" w p ls 1;
plot "failcase-sc-0.80.res" t "D > B" w p ls 1;
plot "failcase-sc-0.90.res" t "D > B" w p ls 1;
plot "failcase-sc-0.99.res" t "D > B" w p ls 1;

unset multiplot;