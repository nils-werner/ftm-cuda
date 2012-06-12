M = importdata('bench.csv', ';', 1);

tries = 5;
nrblocksizes = 25;
nrfilters = length(M.data)/nrblocksizes;

turnaround = 4;
roundtrip = 5;
overall = 6;

query = inline('find(ismember(M.textdata(:,col), search)==1)-1','M','col','search');
get = inline('M.data(val,[1 2 col])','M','col','val');

gpugpu = intersect(query(M,1,'gpu'), query(M,2,'gpu'));
gpucpu = intersect(query(M,1,'gpu'), query(M,2,'cpu'));
cpugpu = intersect(query(M,1,'cpu'), query(M,2,'gpu'));
cpucpu = intersect(query(M,1,'cpu'), query(M,2,'cpu'));

x = get(M, overall, gpugpu)
x = blkproc(x, [tries 1], @mean)
x = reshape(x(:,3), nrblocksizes, [])
surf(x)