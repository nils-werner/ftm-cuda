% M = importdata('bench.csv', ';', 1);

turnaround = 4;
roundtrip = 5;
overall = 6;

query = inline('find(ismember(M.textdata(:,col), search)==1)','M','col','search');
get = inline('M.data(val,col)','M','col','val');

gpugpu = intersect(query(M,1,'gpu'), query(M,2,'gpu'));
gpucpu = intersect(query(M,1,'gpu'), query(M,2,'cpu'));
cpugpu = intersect(query(M,1,'cpu'), query(M,2,'gpu'));
cpucpu = intersect(query(M,1,'cpu'), query(M,2,'cpu'));

get(M, roundtrip, gpugpu)