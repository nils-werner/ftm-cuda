sig = importdata('sig.dat');
x=1:size(sig,2);

%%
plot(x,sig);

%%

sound(sig./abs(max(sig)), 44100);