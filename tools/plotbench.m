M = importdata('bench.csv', ';', 1);

tries = 5;
nrblocksizes = 25;
nrfilters = length(M.data)/(nrblocksizes*4*tries);

filters = 1;
blocklength = 2;
samples = 3;
turnaround = 4;
roundtrip = 5;
overall = 6;

query = inline('find(ismember(M.textdata(:,col), search)==1)-1','M','col','search');
get = inline('M.data(val,[1 2 col])','M','col','val');

gpugpu = intersect(query(M,1,'gpu'), query(M,2,'gpu'));
gpucpu = intersect(query(M,1,'gpu'), query(M,2,'cpu'));
cpugpu = intersect(query(M,1,'cpu'), query(M,2,'gpu'));
cpucpu = intersect(query(M,1,'cpu'), query(M,2,'cpu'));

z = get(M, overall, gpugpu);
z = blkproc(z, [tries 1], @mean);

x = z(:,1);
y = z(:,2);
z = z(:,3)

x = reshape(x, [], nrfilters);
x = x(1,:)';
y = reshape(y, nrblocksizes, []);
y = y(:,1);
z = reshape(z, nrblocksizes, []);

surf(x,y,z)

xlabel('Filters');
ylabel('Blocksize');