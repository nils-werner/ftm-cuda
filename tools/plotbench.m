clear;

files = dir('bench/*.csv');
files = files(end-4:end);
question = '';

for i = 5:-1:1
    question = strcat(question, sprintf(' %d: %s\\n', 6-i, files(i).name));
end

in_file = input(strcat(question, 'What file do you want to load? [1] '));
if isempty(in_file)
    in_file = 1;
end

filename = files(6 - in_file).name;

M = importdata(strcat('bench/', filename), ';', 1);

filters = 1;
blocksize = 2;
matrixblocksize = 3;
chunksize = 4;
samples = 5;
turnaround = 6;
roundtrip = 7;
overall = 8;

nr_types = length(unique(strcat(M.textdata(2:end,1),M.textdata(2:end,2))));
nr_filters = length(unique(M.data(:,filters)));
nr_blocksizes = length(unique(M.data(:,blocksize)));
nr_matrixblocksizes = length(unique(M.data(:,matrixblocksize)));
nr_chunksizes = length(unique(M.data(:,chunksize)));
nr_tries = length(M.data(:,3)) / (nr_chunksizes*nr_matrixblocksizes*nr_blocksizes*nr_filters*nr_types);
samplerate = 44100;

query = inline('find(ismember(M.textdata(:,col), search)==1)-1','M','col','search');
get = inline('M.data(val,[1 2 3 4 col])','M','col','val');

gpugpu = intersect(query(M,1,'gpu'), query(M,2,'gpu'));
gpucpu = intersect(query(M,1,'gpu'), query(M,2,'cpu'));
cpugpu = intersect(query(M,1,'cpu'), query(M,2,'gpu'));
cpucpu = intersect(query(M,1,'cpu'), query(M,2,'cpu'));

modes = [gpugpu, cpucpu, cpugpu, gpucpu];
timers = [roundtrip, turnaround];

in_mode = input(' 1: GPU/GPU\n 2: CPU/CPU\n 3: CPU/GPU\n 4: GPU/CPU\nWhat data do you want to load? [1] ');
if isempty(in_mode) || in_mode > 4
    in_mode = 1;
end

in_timer = input(' 1: Roundtrip\n 2: Turnaround\nWhat timer do you want to load? [1] ');
if isempty(in_timer) || in_timer > 2
    in_timer = 1;
end


z = get(M, timers(in_timer), modes(:,in_mode));
z = blkproc(z, [nr_tries 1], @mean);
z = z(:,5);
z(find(z == 0)) = NaN;


% chunksize -> filters -> matrixblocklength -> blocksize

% filters
v = unique(M.data(:,filters));

% blocksize
w = unique(M.data(:,blocksize));

% matrixblocksize
x = unique(M.data(:,matrixblocksize));

% chunksize
y = unique(M.data(:,chunksize));

z = reshape(z, nr_chunksizes, nr_filters, nr_matrixblocksizes, []);
z = z/1000000;

labels = ['f' 'b' 'm' 'c'];
axistofields = [chunksize filters matrixblocksize blocksize];

question = '';

in_eliminate = zeros(1,2);

in_val = input(sprintf(' 1: Chunksize (%d)\n 2: Filter (%d)\n 3: Matrixblocksize (%d)\n 4: Blocksize (%d)\nWhat variable do you want to eliminate? [1] ', length(y), length(v), length(x), length(w)));
if isempty(in_val) || in_val > 4
    in_val = 1;
end
in_eliminate(1) = in_val;

idx = input(sprintf('What entry do you want to see of the eliminated variable? [1] '));
if isempty(idx)
    idx = 1;
end

in_val = input(sprintf(' 1: Chunksize (%d)\n 2: Filter (%d)\n 3: Matrixblocksize (%d)\n 4: Blocksize (%d)\nWhat variable do you want to eliminate? [2] ', length(y), length(v), length(x), length(w)));
if isempty(in_val) || in_val > 4
    in_val = 2;
end
in_eliminate(2) = in_val;

idy = input(sprintf('What entry do you want to see of the eliminated variable? [1] '));
if isempty(idy)
    idy = 1;
end


% Geschwindigkeit über Blocksize und Chunksize
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(w))])

xy = setdiff([1 2 3 4],in_eliminate);

d = permute(z,[xy in_eliminate]);
d = d(:,:,idx,idy);
%d = permute(d,[3 4 1 2]);

if in_timer == 1 && length(find(axistofields(xy))) > 0
    repmatdim = [1 length(unique(M.data(:,axistofields(xy(2)))))];
    repmatdim = repmatdim';
    divmat = repmat(v, repmatdim);
    if find(axistofields(xy) == 1) == 2
        divmat = divmat';
    end
    d = 1./(d./(divmat./44100));
end

surf(unique(M.data(:,axistofields(xy(2)))),unique(M.data(:,axistofields(xy(1)))),d);
axis vis3d
xlabel(labels(axistofields(xy(2))));
ylabel(labels(axistofields(xy(1))));
if in_timer == 1
    zlabel('v');
else
    zlabel('s');
end
legend(sprintf('%d %s, %d %s',M.data(idy,axistofields(in_eliminate(1))),labels(axistofields(in_eliminate(1))),M.data(idx,axistofields(in_eliminate(2))),labels(axistofields(in_eliminate(2)))));