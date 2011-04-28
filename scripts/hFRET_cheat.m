function save_name = hFRET(save_name,Ktru,FRET,tru_par,prior0)

% Parameter Setting 
%addpath('./src')
K = Ktru+2      

%%
R = 20;
I = 20;
N = length(FRET);
out = cell(R,K);
LP = cell(R,K);
sumc = cell(R,K);
sumv = cell(R,K);
z_hat = cell(R,K);
x_hat = cell(R,K);
u = cell(R,K);
theta = cell(R,K);
Hbest = zeros(1,K);
vb_opts = get_hFRET_vbopts();

kinf = length(prior0.upi);

for r = 1:R
    for k = kinf
        out{r,k} = cell(1,N); LP{r,k} = -inf*ones(1,N);
        z_hat{r,k} = cell(1,N); x_hat{r,k} = cell(1,N);
        
        for n=1:N
                disp(sprintf('r: %d k: %d n:%d',r,k,n))
            for i = 1:I
                if r == 1
                    initM = get_M0(prior0,length(FRET{n}));
                    prior = prior0;
                else
                    initM = get_M0(u{r-1,k},length(FRET{n}));
                    prior = u{r-1,k};
                end
                temp_out = VBEM_eb(FRET{n}, initM, prior,vb_opts);
                % Only save the iterations with the best out.F
                if temp_out.F(end) > LP{r,k}(n)
                    LP{r,k}(n) = temp_out.F(end);
                    out{r,k}{n} = temp_out;
                end
            end 
        end
        
        for n = 1:N
            [z_hat{r,k}{n} x_hat{r,k}{n}] = chmmViterbi_eb(out{r,k}{n},FRET{n});
        end
        % compute posterior hyperparmaters and most probable posterior parameters
        [u{r,k} theta{r,k}] = get_ML_par(out{r,k},prior);

        % check results quality
        [sumc{r,k} sumv{r,k}] = bj_analysis(out{r,k},z_hat{r,k},tru_par);

        save(save_name)
    end
end
 