close all
clear all

%% Flat function landscpae
close all;
hold on;

func = @(x)(-(exp(-x.^2)/2)/(sqrt(2*pi))+1);
g_func = @(x)((2251799813685248*x*exp(-x.^2))/5644425081792261);
h_func = @(x)((2251799813685248*exp(-x.^2))/5644425081792261 - ...
              (4503599627370496*x.^2*exp(-x.^2))/5644425081792261);

xs = -10:0.01:10;
params.it_lim = 100; params.tol = 1e-12; params.init = .5; 
params.g_func = g_func; params.h_func = h_func; params.alpha = .1;

[mins, ~] = nwt_min(params);
plot(xs, func(xs), 'r', 'LineWidth', 2);
for i = 1:length(mins)
    comet(mins(i), func(mins(i)));
    drawnow;
end


%% Smooth function landscape
close all;
hold on;

s_func   = @(x)(exp(-.1*x) + exp(.1*x));
ds_func  = @(x)(exp(x/10)/10 - exp(-x/10)/10);
dds_func = @(x)(exp(-x/10)/100 + exp(x/10)/100);

xs = -10:0.01:10;

params.it_lim = 100; params.tol = 1e-12; params.init = 7; 
params.g_func = ds_func; params.h_func = dds_func; params.alpha = .15;

[mins, ~] = nwt_min(params);
plot(xs, s_func(xs), 'r', 'LineWidth', 2);
for i = 1:length(mins)
    comet(mins(i), s_func(mins(i)));
    drawnow;
end


%% Minima function Newtons method
close all;
hold on;

m_func    = @(x)(sin(10*pi.*x)./(2.*x)+(x-1).^4);
d_m_func  = @(x)(4*(x - 1).^3 - sin(10*pi*x)./(2*x.^2) + (5*pi*cos(10*pi*x))./x);
dd_m_func = @(x)(12*(x - 1).^2 + sin(10*pi*x)./x.^3 - (10*pi*cos(10*pi*x))./x.^2 - (50*pi^2*sin(10*pi*x))./x);
xs = .5:0.01:2.5;
plot(xs, m_func(xs), 'r', 'LineWidth', 2);

params.it_lim = 100; params.tol = 1e-12; params.init = 2.4; 
params.g_func = d_m_func; params.h_func = dd_m_func; params.alpha = .2;

[mins, ~] = nwt_min(params);
for i = 1:length(mins)
    comet(mins(i), m_func(mins(i)));
    drawnow;
end

%% Minima function Gauss-Newtons method
close all;

m_func    = @(x)(sin(10*pi.*x)./(2.*x)+(x-1).^4);
d_m_func  = @(x)(4*(x - 1).^3 - sin(10*pi*x)./(2*x.^2) + (5*pi*cos(10*pi*x))./x);
dd_m_func = @(x)(12*(x - 1).^2 + sin(10*pi*x)./x.^3 - (10*pi*cos(10*pi*x))./x.^2 - (50*pi^2*sin(10*pi*x))./x);
xs = .5:0.01:2.5;

params.it_lim = 500; params.tol = 1e-6; params.init = 1.4; 
params.g_func = d_m_func; params.h_func = dd_m_func; params.alpha = .05;
[mins, hessians] = gn_min(params);

fig = figure();
subplot(1, 3, 1)
plot(xs, m_func(xs), 'r', 'LineWidth', 2);
hold on;

for i = 1:length(mins)
    fprintf("Value: %f, Hess: %f\n", mins(i), hessians(i));
    subplot(1,3, 1);
    plot(mins(i), m_func(mins(i)), 'b*');
    subplot(1,3, 2);
    bar(i, mins(i), 'r');
    subplot(1,3, 3);
    bar(i, hessians(i), 'g');
    drawnow;
end

%% Minima sampling with newton method
close all
clear all
hold on

m_func    = @(x)(sin(10*pi.*x)./(2.*x)+(x-1).^4);
d_m_func  = @(x)(4*(x - 1).^3 - sin(10*pi*x)./(2*x.^2) + (5*pi*cos(10*pi*x))./x);
dd_m_func = @(x)(12*(x - 1).^2 + sin(10*pi*x)./x.^3 - (10*pi*cos(10*pi*x))./x.^2 - (50*pi^2*sin(10*pi*x))./x);
xs = -1:0.01:2.5;
plot(xs, m_func(xs), 'r', 'LineWidth', 2);

params.it_lim = 100; params.tol = 1e-12; params.init = 1.57; 
params.g_func = d_m_func; params.h_func = dd_m_func; params.alpha = .9;

[mins, hess] = nwt_min(params);
nwt_args.hess = hess(end); nwt_args.sol = mins(end);

s_params.cov = 10; s_params.func = m_func;
s_params.imp = 0; s_params.it = 10;
s_params.samps = 100; s_params.lambda = .3;
s_params.input = params.init;

% for it = 1:length(mins)
%    comet(mins(it), m_func(mins(it)));
%    drawnow;
% end

% nwt_args.hess = hess(end); nwt_args.sol = mins(end);
% pi_mins = pi_sampling(s_params, nwt_args);

while true
    [mins, hess] = nwt_min(params);
    nwt_args.hess = .01* hess(end); nwt_args.sol = mins(end);
    pi_mins = pi_sampling(s_params, nwt_args);
    comet(pi_mins(end), m_func(pi_mins(end)));
    s_params.input = pi_mins(end);
    params.init = s_params.input;
    drawnow;
end

%% Minima sampling with gauss newton
close all
clear all
hold on

m_func    = @(x)(sin(10*pi.*x)./(2.*x)+(x-1).^4);
d_m_func  = @(x)(4*(x - 1).^3 - sin(10*pi*x)./(2*x.^2) + (5*pi*cos(10*pi*x))./x);
dd_m_func = @(x)(12*(x - 1).^2 + sin(10*pi*x)./x.^3 - (10*pi*cos(10*pi*x))./x.^2 - (50*pi^2*sin(10*pi*x))./x);
xs = -1:0.01:2.5;
plot(xs, m_func(xs), 'r', 'LineWidth', 2);

params.it_lim = 10; params.tol = 1e-6; params.init = 1.4; 
params.g_func = d_m_func; params.h_func = dd_m_func; params.alpha = .15;

s_params.cov = .5; s_params.func = m_func;
s_params.imp = 0; s_params.it = 10;
s_params.samps = 100; s_params.lambda = .1;
s_params.input = params.init;

while true
    [mins, hess] = gn_min(params);
    nwt_args.hess = 1* hess(end); nwt_args.sol = mins(end);
    pi_mins = pi_sampling(s_params, nwt_args);
    comet(pi_mins(end), m_func(pi_mins(end)));
    s_params.input = pi_mins(end);
    params.init = s_params.input;
    drawnow;
end



%% Flat function landscpae
close all;
clear all;
hold on;

m_func = @(x)(-(exp(-x.^2)/2)/(sqrt(2*pi))+1);
d_m_func = @(x)((2251799813685248*x*exp(-x.^2))/5644425081792261);
dd_m_func = @(x)((2251799813685248*exp(-x.^2))/5644425081792261 - ...
              (4503599627370496*x.^2*exp(-x.^2))/5644425081792261);

xs = -10:0.01:10;
plot(xs, m_func(xs), 'r', 'LineWidth', 2);

params.it_lim = 1; params.tol = 1e-6; params.init = -1; 
params.g_func = d_m_func; params.alpha = .15;

s_params.cov = 1; s_params.func = m_func;
s_params.imp = 1; s_params.it = 1;
s_params.samps = 4; s_params.lambda = .1;
s_params.input = params.init; s_params.hess_reg = 1;

while true
    [mins, hess] = gn_min(params);
    nwt_args.hess = hess(end); nwt_args.sol = mins(end);
    pi_mins = pi_sampling(s_params, nwt_args);
    comet(pi_mins(end), m_func(pi_mins(end)));
    plot(mins(end), m_func(mins(end)), 'b*');
    s_params.input = pi_mins(end);
    params.init = s_params.input;
    drawnow;
   pause(1);
end


%% Flat function GaussNewton
clear all;
close all;
hold on;

func = @(x)(-(exp(-x.^2)/2)/(sqrt(2*pi))+1);
g_func = @(x)((2251799813685248*x*exp(-x.^2))/5644425081792261);
h_func = @(x)((2251799813685248*exp(-x.^2))/5644425081792261 - ...
              (4503599627370496*x.^2*exp(-x.^2))/5644425081792261);

xs = -10:0.01:10;
params.it_lim = 50; params.tol = 1e-6; params.init = -2; 
params.g_func = g_func; params.alpha = .15;

[mins, ~] = gn_min(params);
plot(xs, func(xs), 'r', 'LineWidth', 2);
for i = 1:length(mins)
    comet(mins(i), func(mins(i)));
    drawnow;
end

%% Flat function path integral
close all;
clear all;
hold on;

func = @(x)(-(exp(-x.^2)/2)/(sqrt(2*pi))+1);
g_func = @(x)((2251799813685248*x*exp(-x.^2))/5644425081792261);
h_func = @(x)((2251799813685248*exp(-x.^2))/5644425081792261 - ...
              (4503599627370496*x.^2*exp(-x.^2))/5644425081792261);

xs = -10:0.01:10;
s_params.cov = .15; s_params.func = func;
s_params.imp = 0; s_params.it = 10;
s_params.samps = 100; s_params.lambda = .1;
s_params.input = -8; s_params.hess_reg = 1;

nwt_args.hess = 1; nwt_args.sol = 1;
mins = pi_sampling(s_params, nwt_args);

plot(xs, func(xs), 'r', 'LineWidth', 2);
for i = 1:length(mins)
    comet(mins(i), func(mins(i)));
    drawnow;
end
