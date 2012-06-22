clear;

M = importdata('bench/bench-120621-1615-kleinere-bloecke.csv', ';', 1);

nr_types = 1; % gpu/gpu, cpu/cpu etc.
nr_tries = 3;
nr_blocksizes = 8;
nr_chunksizes = 8;
nr_filters = length(M.data)/(nr_blocksizes*nr_chunksizes*nr_types*nr_tries);
samplerate = 44100;

filters = 1;
blocksize = 2;
blocklength = 3;
samples = 4;
turnaround = 5;
roundtrip = 6;
overall = 7;

query = inline('find(ismember(M.textdata(:,col), search)==1)-1','M','col','search');
get = inline('M.data(val,[1 2 3 col])','M','col','val');

gpugpu = intersect(query(M,1,'gpu'), query(M,2,'gpu'));
gpucpu = intersect(query(M,1,'gpu'), query(M,2,'cpu'));
cpugpu = intersect(query(M,1,'cpu'), query(M,2,'gpu'));
cpucpu = intersect(query(M,1,'cpu'), query(M,2,'cpu'));

%%

z = get(M, roundtrip, gpugpu);

%z = get(M, turnaround, gpugpu);
z = blkproc(z, [nr_tries 1], @mean);

w = z(:,1);
x = z(:,2);
y = z(:,3);
z = z(:,4);

% chunksize -> filters -> blocklength

% filters
w = reshape(w, nr_chunksizes, nr_filters, []);
w = w(1,:,1);
w = permute(w,[2 1 3]);

% blocksize
x = reshape(x, nr_chunksizes, nr_filters, []);
x = x(1,1,:);
x = permute(x,[3 2 1]);

% chunksize
y = reshape(y, nr_chunksizes, nr_filters, []);
y = y(:,1,1);
y = permute(y,[1 3 2]);

z = reshape(z, nr_chunksizes, nr_filters, []);
z = z/1000000;

%%
idx = 45;

v = z(:,idx,:);
v = permute(v,[1 3 2]);

v = 1./(v./(repmat(y, 1, length(x))./44100));

surf(x,y,v);
xlabel('Blocksize');
ylabel('Chunksize');
zlabel('Vielfache der Wiedergabegeschwindigkeit');


%%
idx = 1;

v = z(:,:,idx);
v = permute(v,[1 2 3]);

v = 1./(v./(repmat(y, 1, length(w))./44100));

surf(w(10:20),y,v(:,10:20))
xlabel('Filter');
ylabel('Chunksize');
zlabel('Vielfache der Wiedergabegeschwindigkeit');


%%


idx = 4;

v = z(idx,:,:);
v = permute(v,[3 2 1]);
v =  1./(v./y(idx)*44100);

surf(w,x,v)
xlabel('Filter');
ylabel('Blocksize');
zlabel('Vielfache der Wiedergabegeschwindigkeit');

%%










xlabel('Filter');
ylabel('Blockgroesse');
zlabel('Sekunden');
zlabel('Vielfache der Wiedergabegeschwindigkeit');

%%

x = x(10);
z = z(:,10);

plot(y,z);
ylabel('Vielfache der Wiedergabegeschwindigkeit');
xlabel('Blockgroesse');