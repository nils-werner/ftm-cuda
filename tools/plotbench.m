clear;

figure;

M = importdata('bench/bench-120621-1615-kleinere-bloecke.csv', ';', 1);

types = 1; % gpu/gpu, cpu/cpu etc.
tries = 3;
nrblocklengths = 8;
nrblocksizes = 8;
nrfilters = length(M.data)/(nrblocklengths*nrblocksizes*types*tries);
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
z = blkproc(z, [tries 1], @mean);

w = z(:,1);
x = z(:,2);
y = z(:,3);
z = z(:,4);

% chunksize -> filters -> blocklength

% filters
w = reshape(w, nrblocklengths, nrfilters, []);
w = w(1,:,1);
w = permute(w,[2 1 3]);

% blocksize
x = reshape(x, nrblocklengths, nrfilters, []);
x = x(1,1,:);
x = permute(x,[3 2 1]);

% chunksize
y = reshape(y, nrblocklengths, nrfilters, []);
y = y(:,1,1);
y = permute(y,[1 3 2]);

z = reshape(z, nrblocklengths, nrfilters, []);
z = z/1000000;

%%

v = z(:,4,:);
v = permute(v,[1 3 2]);

v = 1./(v./(repmat(y, 1, length(x))./44100));

surf(x,y,v);
xlabel('Blocksize');
ylabel('Chunksize');
zlabel('Sekunden');

%%

v = z(:,:,1);
v = permute(v,[1 2 3]);

v = 1./(v./(repmat(y, 1, length(w))./44100));

surf(w,y,v)
xlabel('Filter');
ylabel('Chunksize');
zlabel('Sekunden');

%%

v = z(:,:,1);
v = permute(v,[1 2 3]);

v = 1./(v./(repmat(y, 1, length(w))./44100));

surf(w(16:32),y,v(:,16:32))
xlabel('Filter');
ylabel('Chunksize');
zlabel('Sekunden');


%%

v = z(4,:,:);
v = permute(v,[3 2 1]);
v =  1./(v./y(4)./44100);

surf(w,x,v)
xlabel('Filter');
ylabel('Blocksize');
zlabel('Sekunden');

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