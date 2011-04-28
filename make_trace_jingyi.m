clear;clc;close all;
%%%%%%%%%%%%%%%%
saveout = 1;
plotPlots = 1;

N=300;

dname = sprintf('jingyi_40_100');
mus1D = [0.4 0.6];
K = length(mus1D);

Q4 = exp(log(0.5)/40);
%Q4 = exp(log(0.5)/40);
% Q4 = exp(log(0.5)/200);
Q6 = exp(log(0.5)/100);
%Q6 = exp(log(0.5)/100);
% Q6 = exp(log(0.5)/300);

A0 = [Q4 1-Q4;1-Q6 Q6];

raw = cell(1,N);
FRET = cell(1,N);
zPath = cell(1,N);
xPath = cell(1,N);
mus = zeros(2,length(mus1D));


for i = 1:K;
    mus(2,i) = 1000*mus1D(i);
    mus(1,i) = 1000*(1-mus1D(i));
end
    

for n = 1:N
    % make standard synthetic traces
    T = round(200);
    covs = get_params_01(mus,25);
    %z0 = ceil(rand*K);
    z0 = 1 + round(rand);
    [raw{n},ig,zPath{n}] = generateCHMC_01...
        (z0,A0,mus,covs,T);

    % transform to 1D fret
    FRET{n} = (raw{n}(:,2)./sum(raw{n},2))';
    xPath{n} = mus1D(zPath{n})';
    
    %photobleach after 80+/10
    Tp = 100 + round(10*randn);
    
    FRET{n}(end-Tp:end) = 0.1;
    xPath{n}(end-Tp:end) = 0.1;


end

data = xPath;
path = xPath;

for n=1:N
    data{n} = [1-data{n}(:) data{n}(:)];
end


% calculate tm;
Aobs = zeros(3);
mus = [0.4 0.6 0.1];
for n=1:length(path)
    zpath=path{n};
    for k=1:length(mus)
        zpath(zpath==mus(k))=k;
    end
    
    for t=1:length(zpath)-1
        Aobs(zpath(t),zpath(t+1)) = Aobs(zpath(t),zpath(t+1)) + 1;
    end
end
Aobs = normalise(Aobs,2)


Aobs2 = zeros(2);
mus = [0.4 0.6];
for n=1:length(path)
    zpath=path{n};
    for k=1:length(mus)
        zpath(zpath==mus(k))=k;
    end
    
    for t=1:length(zpath)-1
        if zpath(t+1) ~= 0.1
            Aobs2(zpath(t),zpath(t+1)) = Aobs2(zpath(t),zpath(t+1)) + 1;
        else
            break
        end
    end
end

Aobs2 = normalise(Aobs2,2)

Aobs = Aobs2;

rate = -log(diag(Aobs))

half_life = -log(0.5)./rate

save(dname,'path','data','A0','mus1D','rate','half_life','Aobs')
