function vit = scaleVitBack(vit, scale)
% Use scale parameter I = mx+ b to scale intensity back.
% scale: [m, b]

for iMol = 1:length(vit)
    vit(iMol).x = vit(iMol).x*scale(iMol, 1) + scale(iMol, 2);
end

end
