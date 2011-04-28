clear;clc;
%%%%%%%%%%%%%%%%
saveout = 1;
plotPlots = 1;

dname = 'randMu_traces_100_cov25';

% keep track of how much means were allowed to fluctuate
Fvec = 0:0.05:0.5;


N=100;F=length(Fvec);

% trace parameters
mus1D = [0.25 0.5 0.75];
K = length(mus1D);
% transition matrix
A0 = ones(K) + 18*diag(ones(1,K))+triu(0.5*ones(K))+3*rand(3);
% covariance multiplier
scale_cov = 25;

Fmtx = zeros(F,N);
for f=1:F
    Fmtx(f,:) = Fvec(f)*randn(1,N);
    
    % make sure distribution of means is actually close to what it should
    % be
    while abs(std(Fmtx(f,:)) - Fvec(f))/Fvec(f) > 0.01
        Fmtx(f,:) = Fvec(f)*randn(1,N);
    end
end
% to correct for the fact that hFRET would otherwise start with guessed
% states directly on top of the hidden states
shift_const = 0.1;
Fmtx = Fmtx + shift_const;


covs_cell = cell(F,N);
raw = cell(F,N);
FRET = cell(F,N);
zPath = cell(F,N);
xPath = cell(F,N);
mus = zeros(2,length(mus1D));


Aobs = cell(F,N);
mu_obs = zeros(F,N,K);
sigs_obs = zeros(F,N,K);


for i = 1:K;
    mus(2,i) = 1000*mus1D(i);
    mus(1,i) = 1000*(1-mus1D(i));
end
    

for f=1:F
    disp(f)
    for n = 1:N
        % make standard synthetic traces
        T = round(50 + rand*450);
        covs_cell{f,n} = get_params_01(mus,scale_cov);
        z0 = ceil(rand*K);
        [raw{f,n},Aobs{f,n},zPath{f,n}] = generateCHMC_01...
            (z0,A0,mus,covs_cell{f,n},T);
        
        % transform to 1D fret
        FRET{f,n} = (raw{f,n}(:,2)./sum(raw{f,n},2))';
        
        % displace each trace by 0.1 + some gaussian noise
        % the 0.1 is to keep the traces different than the starting states
        % used in the hFRET search
        % the gaussian noise is same for each state to avoide states
        % crossing
        
        FRET{f,n} = FRET{f,n} + Fmtx(f,n);
        xPath{f,n} = mus1D(zPath{f,n}) + Fmtx(f,n);

        for k = 1:K
            mu_obs(f,n,k) = mean(FRET{f,n}(zPath{f,n} == k));
            sigs_obs(f,n,k) = std(FRET{f,n}(zPath{f,n} == k));
        end
    end
end

% plot traces
if plotPlots
    count = 0;
    for f=1:F
        for n = 1:9                
            count = count + 1;
            T = length(FRET{f,n});
            if mod(count,9) == 1
                figure
            end
            subplot(3,3,1+mod(8+count,9));
            plot(FRET{f,n})
            hold on
            plot(xPath{f,n},'k')    
            set(gca,'YTick',0:0.2:1)
    %                 axis([0 T+1 -.2 1]);
            title(sprintf('f%d n%d',f,n))
        end
    end
end

% make file name
d_t = clock;


if saveout
    save(dname)
end
