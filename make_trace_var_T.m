% make sets of traces with same total numbers of time points, but spread
% over more and more traces (i.e. trace length getting smaller).
clear;clc;
%%%%%%%%%%%%%%%%
saveout = 1;
plotPlots = 0;


%total number of data points per set
Ttot = 20000;

% trace size
Tvec = [500 250 150 100 75 50 25 10]

% Fluctuation around mean
F = 0.1;

L = length(Tvec);

% length of traces in each data set
Nvec = round(Ttot ./ Tvec);

% trace parameters
mus1D = [0.25 0.5 0.75];

K = length(mus1D);
% transition matrix
A0 = normalise(ones(K)+3*triu(ones(K))+3*rand(3)+diag([80 30 5]),2);
% covariance multiplier
scale_cov = 40;

dname = sprintf('smallT_traces_20k_c%d_020110',scale_cov);



Fcell = cell(1,L);
for l=1:L
    N = Nvec(l);
    Fcell{l} = F*randn(1,N);
    % make sure distribution of means is actually close to what it should
    % be
    while abs(std(Fcell{l}) - F)/F > 0.01
        Fcell{l} = F*randn(1,N);
    end
end

% to correct for the fact that hFRET would otherwise start with guessed
% states directly on top of the hidden states
shift_const = 0.1;
for l=1:L
    Fcell{l} = Fcell{l} + shift_const;
end

covs_cell = cell(1,L);
raw = cell(1,L);
FRET = cell(1,L);
zPath = cell(1,L);
xPath = cell(1,L);
mus = zeros(2,length(mus1D));


Aobs = cell(1,L);
mu_obs = cell(1,L);
sigs_obs = cell(1,L);
for l=1:L
    N=Nvec(l);
    mu_obs{l} = zeros(N,K);
    sig_obs{l} = zeros(N,K);
    
    Aobs{l} = cell(1,N);
    covs_cell{l} = cell(1,N);
    raw{l} = cell(1,N);
    FRET{l} = cell(1,N);
    zPath{l} = cell(1,N);
    xPath{l} = cell(1,N);
end

for i = 1:K;
    mus(2,i) = 1000*mus1D(i);
    mus(1,i) = 1000*(1-mus1D(i));
end
    

for l=1:L
    disp(l)
    N=Nvec(l);
    
    for n = 1:N
        % make standard synthetic traces
        T = Tvec(l);
        covs_cell{l}{n} = get_params_01(mus,scale_cov);
        z0 = ceil(rand*K);
        [raw{l}{n},Aobs{l}{n},zPath{l}{n}] = generateCHMC_01...
            (z0,A0,mus,covs_cell{l}{n},T);
        
        % transform to 1D fret
        FRET{l}{n} = (raw{l}{n}(:,2)./sum(raw{l}{n},2))';
        
        % displace each trace by 0.1 + some gaussian noise
        % the 0.1 is to keep the traces different than the starting states
        % used in the hFRET search
        % the gaussian noise is same for each state to avoide states
        % crossing
        
        FRET{l}{n} = FRET{l}{n} + Fcell{l}(n);
        xPath{l}{n} = mus1D(zPath{l}{n}) + Fcell{l}(n);

        for k = 1:K
            mu_obs{l}(n,k) = mean(FRET{l}{n}(zPath{l}{n} == k));
            sigs_obs{l}(n,k) = std(FRET{l}{n}(zPath{l}{n} == k));
        end
    end
end

% plot traces
if plotPlots
    count = 0;
    for l=1:L
        for n = 1:9                
            count = count + 1;
            T = length(FRET{l}{n});
            if mod(count,9) == 1
                figure
            end
            subplot(3,3,1+mod(8+count,9));
            plot(FRET{l}{n})
            hold on
            plot(xPath{l}{n},'k')    
            set(gca,'YTick',0:0.2:1)
    %                 axis([0 T+1 -.2 1]);
            title(sprintf('l%d n%d',l,n))
        end
    end
end

% make file name
d_t = clock;


if saveout
    save(dname)
end
