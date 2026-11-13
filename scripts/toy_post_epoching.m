% Run this script with the repository's scripts directory as current folder.
addpath(fullfile(pwd, 'MatlabPlugin'));
fs = 10000;
times = (-1000:2499)' / fs;  % seconds relative to the TMS pulse at zero
u = max(times - 0.010, 0);
neural = 5 * exp(-25 * u) .* sin(2*pi*50*u);
artifact = -100 * (1 - exp(-u / 0.003));
decay = times >= 0.015;
artifact(decay) = -100 * (1-exp(-0.005/0.003)) * exp(-(times(decay)-0.015)/0.200);
data = reshape(neural + artifact, 1, 1, []);  % epochs x channels x samples

cfg = tmspath_config(); cfg.channel_policy='all'; cfg.fill_mode='none';
cfg.onset_s = 0.010; cfg.peak_s = 0.015; cfg.end_s = times(end);
cfg.rise_model = 'lagrange'; cfg.decay_model = 'poly';
cfg.rise_degree = 3; cfg.decay_degree = 3;
cfg.constrain_rise = false; cfg.constrain_decay = false; cfg.pre_degree = 2;
[corrected, trend, diagnostics] = tmspath_detrend(data, times, cfg);
mask = times >= cfg.onset_s;
recovered = reshape(corrected, [], 1);
mse = mean((recovered(mask) - neural(mask)).^2);
assert(isequal(size(corrected), size(data)));
assert(max(abs(corrected(:)+trend(:)-data(:)))<1e-12);
fprintf('Toy recovery MSE: %.6f a.u.^2\n', mse);

out = fullfile(pwd, 'images'); if ~exist(out,'dir'), mkdir(out); end
fig = figure('Color','w','Visible','off','Position',[100 100 1000 700]);
tl = tiledlayout(fig,2,1,'TileSpacing','compact');
ax1 = nexttile(tl);
plot(times*1000, reshape(data,[],1), 'DisplayName','Input: neural + slow artifact'); hold on;
plot(times*1000, reshape(trend,[],1), '--', 'DisplayName','Estimated trend');
ax2 = nexttile(tl);
plot(times*1000, neural, 'k', 'DisplayName','Known toy neural signal'); hold on;
plot(times*1000, recovered, 'DisplayName','Polynomial');
for ax = [ax1 ax2]
    xline(ax,10,':','DisplayName','Detrend onset: 10 ms');
    ylabel(ax,'Amplitude (a.u.)'); xlim(ax,[-20 250]); legend(ax,'Location','best');
end
xlabel(ax2,'Time relative to TMS pulse (ms)');
title(tl,sprintf('MATLAB post-epoching toy - Lagrange rise / cubic decay - MSE %.4f a.u.^2',mse));
exportgraphics(fig,fullfile(out,'toy_post_epoching_matlab.png'),'Resolution',160);close(fig);
results = fullfile(pwd,'results','toy'); if ~exist(results,'dir'), mkdir(results); end
save(fullfile(results,'matlab_toy.mat'),'times','data','neural','corrected','trend','mse');
% Pass corrected, or the EEGLAB adapter output, to subsequent ICA in real data.
