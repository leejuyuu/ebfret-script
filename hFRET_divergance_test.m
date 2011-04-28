% Parameter Setting 
addpath('./src')
clc;clear;
load hFRET_test

fret = 0.15*randn(50,1) + 0.5;
%fret2 = 0.15*randn(500,1) + 0.5;
vb_opts = get_hFRET_vbopts();

priors{1}.W = 4.4;
priors{1}.v = 10;
priors{1}.beta = 1;
priors{1}.mu = 0.5;
priors{1}.upi = 1;
priors{1}.ua = 1;        

priors{2}.W = [4.4 4.4];
priors{2}.v = [10 10]';
priors{2}.beta = [1 1]';
priors{2}.mu = [0.3 0.7];
priors{2}.upi = [1 1];
priors{2}.ua = ones(2);


priors{3}.W = [100000 4.4];
priors{3}.v = [100000 10]';
priors{3}.beta = [1 1]';
priors{3}.mu = [fret(2) 0.5];
priors{3}.upi = [1 1];
priors{3}.ua = ones(2);

H = 3; N = 1;
I = 20;

out = cell(1,H);
LP = cell(1,H);
z_hat = cell(1,H);
x_hat = cell(1,H);
u = cell(1,H);
theta = cell(1,H);

fret = fret(:)';

for h =  1:H
    u0 = priors{h};
    LP{h} = -inf;
    for i = 1:I
        initM = get_M0(u0,length(fret));
        temp_out = VBEM_eb(fret, initM, u0,vb_opts);
        % Only save the iterations with the best out.F
        if temp_out.F(end) > LP{h}
            LP{h} = temp_out.F(end);
            out{h} = temp_out;
        end
    end

    [z_hat{h} x_hat{h}] = chmmViterbi_eb(out{h},fret);
end

figure
for h=1:3
subplot(3,1,h)
hold on
plot(fret);
plot(x_hat{h},'r')
title(sprintf('%f',LP{h}))
end