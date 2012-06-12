set term postscript eps color blacktext "Helvetica" 24
set output 'test.eps'
set label 1 "This is the surface boundary" at -10, -5, 150 centre norotate back nopoint offset character 0, 0, 0
set arrow 1 from -10, -5, 120 to -10, 0, 0 nohead back nofilled linetype -1 linewidth 1.000
set arrow 2 from -10, -5, 120 to 10, 0, 0 nohead back nofilled linetype -1 linewidth 1.000
set arrow 3 from -10, -5, 120 to 0, 10, 0 nohead back nofilled linetype -1 linewidth 1.000
set arrow 4 from -10, -5, 120 to 0, -10, 0 nohead back nofilled linetype -1 linewidth 1.000
set samples 21, 21
set isosamples 11, 11
set title "3D gnuplot demo" 
set xlabel "X axis" 
set xlabel  offset character -3, -2, 0 font "" textcolor lt -1 norotate
set xrange [ -10.0000 : 10.0000 ] noreverse nowriteback
set ylabel "Y axis" 
set ylabel  offset character 3, -2, 0 font "" textcolor lt -1 rotate by -270
set yrange [ -10.0000 : 10.0000 ] noreverse nowriteback
set zlabel "Z axis" 
set zlabel  offset character -5, 0, 0 font "" textcolor lt -1 norotate
splot x*y
