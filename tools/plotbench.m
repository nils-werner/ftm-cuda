clear;

M = importdata('bench/bench-120628-1214-abbildung-6-6.csv', ';', 1);
%M = importdata('bench/bench-120620-0000-all-nach-blockopt.csv', ';', 1);

nr_types = length(unique(M.textdata(2:end,1))) * length(unique(M.textdata(2:end,2)));
nr_filters = length(unique(M.data(:,1)));
nr_blocksizes = length(unique(M.data(:,2)));
nr_chunksizes = length(unique(M.data(:,3)));
nr_tries = length(M.data(:,3)) / (nr_chunksizes*nr_blocksizes*nr_filters*nr_types);
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


z = get(M, roundtrip, gpugpu);
%z = get(M, roundtrip, cpucpu);

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

idx = 1;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(w))])

v = z(:,idx,:);
v = permute(v,[1 3 2]);

v = 1./(v./(repmat(y, 1, length(x))./44100));

surf(x,y,v);
axis vis3d
xlabel('b');
ylabel('c');
zlabel('v');
legend(sprintf('Filter %d', w(idx)));

%%

idx = 1;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(x))])

v = z(:,:,idx);
v = permute(v,[1 2 3]);

v = 1./(v./(repmat(y, 1, length(w))./44100));

surf(w,y,v)
axis vis3d
xlabel('f');
ylabel('c');
zlabel('v');
legend(sprintf('Blockgroesse %d', x(idx)));

%%

idx = 1;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(y))])

v = z(idx,:,:);
v = permute(v,[3 2 1]);

v =  1./(v./y(idx)*44100);

surf(w,x,v)
axis vis3d
xlabel('f');
ylabel('b');
zlabel('v');
legend(sprintf('Chunkgroesse %d', y(idx)));

%%

idx = 1;
idy = 1;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(x))])

v = z(:,idy,idx);
v = permute(v,[1 2 3]);

v = 1./(v./y.*44100);

plot(y,v)
axis vis3d
xlabel('c');
ylabel('v');
legend(sprintf('Filter %d', w(idy)));

%%

hold on
idx = 1;
idy = 22;
%idy = 19;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(x))])

v = z(idy,:,idx);
v = permute(v,[1 2 3]);

v = 1./(v./y(idy).*44100);

plot(w,v)
axis vis3d
xlabel('f');
ylabel('v');
legend(sprintf('Chunkgroesse %d', y(idy)));