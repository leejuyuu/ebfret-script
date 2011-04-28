function plot_results(out,FRET,title_str,fig_handle)

[z_hat x_hat] = chmmViterbi_eb(out,FRET);

if nargin < 4
    figure
    fig_handle = axes;
end
hold
plot(fig_handle,FRET,'LineWidth',2)
plot(fig_handle,x_hat,'r','LineWidth',2)
title(fig_handle,title_str,'interpreter','none')

% [zeta smooth] = zeta_calc(zPath{n},z_hat1{n},0.1,umu,bestOut{n}.m);
