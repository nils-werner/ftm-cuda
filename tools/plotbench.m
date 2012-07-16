clear;

M = importdata('bench/bench-120716-1903-getrennte-params.csv', ';', 1);
%M = importdata('bench/bench-120620-0000-all-nach-blockopt.csv', ';', 1);



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

%%

question = '';

in_display = input(sprintf(' 0: None\n 1: Filter (%d)\n 2: Blocksize (%d)\n 3: Chunksize (%d)\nWhat variable do you want to select from? [1] ', length(w), length(x), length(y)));
if isempty(in_display) || in_display > 3
    in_display = 1;
end

if (in_display == 1 && length(w) > 1) || (in_display == 2 && length(x) > 1) || (in_display == 3 && length(y) > 1)
    idx = input(sprintf('What entry do you want to see? [1] ', length(w), length(x), length(y)));
    if isempty(idx)
        idx = 1;
    end
else
    idx = 1;
end

if in_display == 1
    % Geschwindigkeit über Blocksize und Chunksize
    disp(['Displaying item ', num2str(idx), ' of ', num2str(length(w))])

    d = z(:,idx,:,:);
    d = permute(d,[1 3 2]);

    if in_timer == 1
        d = 1./(d./(repmat(y, 1, length(x))./44100));
    end

    surf(x,y,d);
    axis vis3d
    xlabel('b');
    ylabel('c');
    zlabel('v');
    legend(sprintf('Filter %d', w(idx)));
elseif in_display == 2
    % Geschwindigkeit über Filter und Chunksize
    disp(['Displaying item ', num2str(idx), ' of ', num2str(length(x))])

    d = z(:,:,idx);
    d = permute(d,[1 2 3]);

    if in_timer == 1
        d = 1./(d./(repmat(y, 1, length(w))./44100));
    end

    surf(w,y,d)
    axis vis3d
    xlabel('f');
    ylabel('c');
    zlabel('v');
    legend(sprintf('Blockgroesse %d', x(idx)));
elseif in_display == 3
    % Geschwindigkeit über Filter und Blocksize
    disp(['Displaying item ', num2str(idx), ' of ', num2str(length(y))])

    d = z(idx,:,:);
    d = permute(d,[3 2 1]);

    if in_timer == 1
        d =  1./(d./y(idx)*44100);
    end

    surf(w,x,d)
    axis vis3d
    xlabel('f');
    ylabel('b');
    zlabel('v');
    legend(sprintf('Chunkgroesse %d', y(idx)));
elseif in_display == 4
    % Geschwindigkeit über Filter und Blocksize
    disp(['Displaying item ', num2str(idx), ' of ', num2str(length(y))])

    d = z(idx,:,:);
    d = permute(d,[3 2 1]);

    if in_timer == 1
        d =  1./(d./y(idx)*44100);
    end

    surf(w,x,d)
    axis vis3d
    xlabel('f');
    ylabel('b');
    zlabel('v');
    legend(sprintf('Chunkgroesse %d', y(idx)));
end

%%
















idx = 1;
idy = 1;
disp(['Displaying item ', num2str(idx), ' of ', num2str(length(x))])

d = z(:,idy,idx);
d = permute(d,[1 2 3]);

if in_timer == 1
    d = 1./(d./y.*44100);
end

plot(y,d)
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

d = z(idy,:,idx);
d = permute(d,[1 2 3]);

if in_timer == 1
    d = 1./(d./y(idy).*44100);
end

plot(w,d)
axis vis3d
xlabel('f');
ylabel('v');
legend(sprintf('Chunkgroesse %d', y(idy)));