% substitutes all NaN values with 0.
function f = nan_to_zero(f)
    f(isnan(f)) = 0;
end
