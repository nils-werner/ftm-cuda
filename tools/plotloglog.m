clear;
close all;


C = importdata('bench/bench-120718-1440-wettrennen-cpu.csv', ';', 1);
G = importdata('bench/bench-120718-1413-wettrennen-gpu.csv', ';', 1);

chunks = C.data(1,4);
x = C.data(:,1);
cpu = C.data(:,7);
gpu = G.data(:,7);

cpu = cpu/1000000;
gpu = gpu/1000000;

cpu = 1./(cpu./(chunks./44100));
gpu = 1./(gpu./(chunks./44100));

close all;
hold on;
plot(x,cpu,'k');
plot(x,gpu,'k--');