function [min_val, hess] = nwt_min(params)
    delta = 1e12; iter = 1;
    min_val = [params.init];
    hess = [params.init];
    while abs(delta) > params.tol && iter < params.it_lim
        delta = -params.g_func(min_val(iter)) * inv(params.h_func(min_val(iter)));
        hess(iter) = params.h_func(min_val(iter));
        min_val(iter+1) = min_val(iter) + delta*params.alpha;
        iter = iter + 1;
    end
    fprintf("** MIN %f at conv level: %f **\n", min_val(end), delta)
end