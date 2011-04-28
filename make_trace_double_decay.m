clear;clc;close all;
%%%%%%%%%%%%%%%%
saveout = 1;
plotPlots = 1;

N=100;

dname = sprintf('double_decay_traces_T600_%d',N);

% keep track of how much means were allowed to fluctuate
Fvec = 0.5:0.05:0.95;
%Fvec = [0.5 0.95]

F=length(Fvec);
A0 = cell(1,F);
% trace parameters
mus1D = [0.3 0.3 0.7 0.7];
K = length(mus1D);
% covariance multiplier
scale_cov = 25;

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
    A0{f} = [Fvec(f) 0 0 1-Fvec(f);0 0.95 0.05 0; 0 0.05 0.95 0; 0.05 0 0 0.95];
    A0{f}
    for n = 1:N
        % make standard synthetic traces
        T = round(600);
        covs_cell{f,n} = get_params_01(mus,scale_cov);
        %z0 = ceil(rand*K);
        z0 = 3 + round(rand);
        z0 = 3+ mod(n,2);
        [raw{f,n},Aobs{f,n},zPath{f,n}] = generateCHMC_01...
            (z0,A0{f},mus,covs_cell{f,n},T);
        
        % transform to 1D fret
        FRET{f,n} = (raw{f,n}(:,2)./sum(raw{f,n},2))';
        
        xPath{f,n} = mus1D(zPath{f,n});

        for k = 1:K
            mu_obs(f,n,k) = mean(FRET{f,n}(zPath{f,n} == k));
            sigs_obs(f,n,k) = std(FRET{f,n}(zPath{f,n} == k));
        end
    end
end


if 0
    for f = 1:F
        for n = 1:10
            figure
            plot(FRET{f,n},'LineWidth',1.5)
            hold on
            plot(xPath{f,n},'k','LineWidth',1.5)    
            set(gca,'YTick',0:0.2:1)
            title_str = sprintf('f%d n%d',f,n)
            title(title_str)
            %saveas(gcf,['~/Desktop/' title_str '.eps'],'psc2')
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
