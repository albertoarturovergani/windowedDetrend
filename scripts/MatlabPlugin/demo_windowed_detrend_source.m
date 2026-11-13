%% Multichannel EEG: threshold first
% Run all sections in order. Synthetic EEG, not experimental validation.
% Base MATLAB only; no plugin installation or extra toolboxes needed.
pluginDir=fileparts(mfilename('fullpath')); addpath(pluginDir);

%% A. Controls in order of importance
cfg=tmspath_config();
% 1. Slope Threshold: main decision parameter for extreme-artifact detection.
% A channel score >= threshold is classified as extreme: enable windowed fits.
% Threshold above ALL channel scores: apply the basic whole-epoch linear fit.
% Scores are absolute epoch-mean standardized slopes. Lower => more flagged.
cfg.slope_threshold=0.5;
% 2. Peak search interval, seconds relative to the recorded TMS trigger.
% The detected peak splits rise/decay; slopes use pre, rise AND decay.
cfg.onset_s=0.0;
cfg.peak_search_end_s=0.020;
% 3. Models, kept at the published Windowed TEP defaults.
cfg.pre_degree=2;
cfg.rise_model='lagrange'; % Interpolates every rise sample; degree N-1.
cfg.decay_model='poly';
cfg.decay_degree=3;       % Offset channels: cubic; others: degree 2.
cfg.fill_mode='gaussian'; % Default: replace zeroed rise plus margin with Gaussian noise.
% All other fields keep their library defaults: channel_policy='auto',
% constrain_decay=false, only_offset_channels=false, overall_degree=1.
% No flagged channel => whole-epoch linear fit, no Lagrange and no filling.
% At least one flagged => windowed fits on ALL selected channels by default.
% Changing cfg does not change the synthetic data below.

%% B. Synthetic EEG: channels, stimuli and jitter
toy=struct('n_epochs',12,'n_channels',16,'fs_hz',2000,'seed',42, ...
 'stimulus_jitter_s',0.0007, ... % Common latency per stimulus: +/-0.7 ms.
 'channel_jitter_s',0.00035, ... % Additional latency per channel: +/-0.35 ms.
 'amplitude_jitter',0.10, ...   % Artifact amplitude varies +/-10% per trace.
 'noise_sd',0.05);             % Independent noise, arbitrary units.
[data,times,truth,artifact,metadata]=tmspath_demo_data(toy);
labels=metadata.channel_labels;
% Two channels have stronger artifacts, with weaker contamination elsewhere.
% A damped response and ongoing background provide known clean ground truth.
% Jitter changes signal latency relative to fixed event markers at t=0;
% it does not resample epochs or introduce the removed pulse artifact.
fprintf('Input shape (epochs, channels, samples): %d x %d x %d\n',size(data));
disp('Stimulus jitter (ms):'); disp(1000*metadata.stimulus_jitter_s');

%% 1. Apply automatic screening and Polynomial detrending
[corrected,trend,diagnostics]=tmspath_detrend(data,times,cfg);
selection=diagnostics.channel_selection;
scores=max(abs(selection.mean_z),[],2);
offset_channels=selection.offset_channels;
operation=diagnostics.operation;
mask=times>=0;
error_array=corrected(:,:,mask)-truth(:,:,mask);
mse_full=mean(error_array(:).^2);
fprintf('Operation: %s\n',operation);
disp('Offset channels:'); disp(labels(offset_channels));
disp('Channel  max|mean z-slope|  classification');
for c=1:toy.n_channels
 classification='below threshold'; if ismember(c,offset_channels), classification='offset'; end
 fprintf('%-7s  %8.4f          %s\n',labels{c},scores(c),classification);
end
fprintf('Post-trigger recovery MSE: %.6f a.u.^2\n',mse_full);
% Inspect operation before accessing rise/decay: fallback records have overall.
if strcmp(operation,'windowed')
 degrees=[];
 for k=1:numel(diagnostics.fits)
  rec=diagnostics.fits{k}; if rec.epoch==1, degrees(end+1)=rec.decay.degree; end %#ok<SAGROW>
 end
 disp('Decay degree by channel:'); disp(degrees);
else
 disp('No windowed detrending: whole-epoch linear fit; no gap filling.');
end

%% 2. Change only the threshold on the SAME data
% This sweep illustrates decisions; it does not optimize the threshold.
sweep_thresholds=unique([.25 .5 1 2 3 cfg.slope_threshold]);
sweep_counts=zeros(size(sweep_thresholds));
sweep_operations=cell(size(sweep_thresholds)); sweep_mse=zeros(size(sweep_thresholds));
disp('Threshold  flagged channels  operation             recovery MSE');
for k=1:numel(sweep_thresholds)
 sweep_cfg=cfg; sweep_cfg.slope_threshold=sweep_thresholds(k);
 [sweep_output,~,sweep_info]=tmspath_detrend(data,times,sweep_cfg);
 sweep_counts(k)=numel(sweep_info.channel_selection.offset_channels);
 sweep_operations{k}=sweep_info.operation;
 error_array=sweep_output(:,:,mask)-truth(:,:,mask);
 sweep_mse(k)=mean(error_array(:).^2);
 fprintf('%8.3f  %16d  %-20s  %.6f\n',sweep_thresholds(k),sweep_counts(k),sweep_operations{k},sweep_mse(k));
end

%% 3. Inspect screening and recovered waveforms
fig=figure('Color','w','Position',[100 100 1200 800]);
layout=tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
blue=[.14 .35 .47]; orange=[.71 .32 .16]; grey=[.46 .54 .60];
ax=nexttile(layout); bars=bar(ax,1:toy.n_channels,scores,'FaceColor','flat');
bars.CData=repmat(grey,toy.n_channels,1);
bars.CData(offset_channels,:)=repmat(orange,numel(offset_channels),1);
hold(ax,'on'); yline(ax,cfg.slope_threshold,'--',sprintf('Threshold = %g',cfg.slope_threshold),'Color',orange);
xticks(ax,1:toy.n_channels); xticklabels(ax,labels); xtickangle(ax,60); ax.FontSize=9;
ylabel(ax,'max |mean slope z-score|'); title(ax,'1. Channel decision');
ax=nexttile(layout);
plot(ax,sweep_thresholds,sweep_counts,'o','Color',blue,'MarkerFaceColor',blue); hold(ax,'on');
for k=1:numel(sweep_thresholds)
 text_label='Windowed'; if strcmp(sweep_operations{k},'overall'), text_label='Linear'; end
 text(ax,sweep_thresholds(k),sweep_counts(k)+.8,text_label,'HorizontalAlignment','center','FontSize',8);
end
xline(ax,cfg.slope_threshold,'--','Color',orange);
ylim(ax,[-1 toy.n_channels+3]); xlim(ax,[min(sweep_thresholds)-.15 max(sweep_thresholds)+.2]);
xlabel(ax,'Slope threshold'); ylabel(ax,'Flagged channels'); title(ax,'2. Same EEG, different threshold');
[~,strong]=max(scores); [~,weak]=min(scores); display_channels=[strong weak];
for c=display_channels
 ax=nexttile(layout);
 y=reshape(truth(1,c,:),[],1); z=reshape(corrected(1,c,:),[],1);
 plot(ax,times*1000,y,'k','DisplayName','Known clean signal'); hold(ax,'on');
 plot(ax,times*1000,z,'Color',orange,'DisplayName','Polynomial');
 limits=ylim(ax); search_ms=1000*[cfg.onset_s cfg.peak_search_end_s];
 patch(ax,search_ms([1 2 2 1]),limits([1 1 2 2]),[.86 .91 .94], ...
  'FaceAlpha',.4,'EdgeColor','none','DisplayName','Peak search');
 subtitle='whole-epoch linear';
 for k=1:numel(diagnostics.fits)
  rec=diagnostics.fits{k};
  if rec.epoch==1 && rec.channel==c && isfield(rec,'peak_s')
   xline(ax,1000*rec.peak_s,':','Color',blue,'DisplayName','Detected split');
   subtitle=sprintf('decay degree %d',rec.decay.degree); break;
  end
 end
 title(ax,sprintf('%s, epoch 1: %s',labels{c},subtitle));
 xlabel(ax,'Time from trigger (ms)'); ylabel(ax,'Amplitude (a.u.)');
 xlim(ax,[-20 200]);
end
% One shared legend in a separate tile, outside both waveform axes.
wave_legend=legend(ax,'Orientation','horizontal','NumColumns',2,'FontSize',9,'Box','off');
wave_legend.Layout.Tile='south';
title(layout,sprintf('Polynomial | %d channels x %d stimuli | %s',toy.n_channels,toy.n_epochs,operation));

%% 4. Inspect the simulated EEG and timing jitter
fig_input=figure('Color','w','Position',[100 100 1200 420]);
input_layout=tiledlayout(fig_input,1,2,'TileSpacing','compact','Padding','compact');
ax=nexttile(input_layout); imagesc(ax,1000*times,1:toy.n_channels,reshape(data(1,:,:),toy.n_channels,[]));
axis(ax,'xy'); xlim(ax,[-20 200]); colorbar(ax);
xlabel(ax,'Time from trigger (ms)'); ylabel(ax,'Channel'); title(ax,'Observed EEG: epoch 1 (a.u.)');
ax=nexttile(input_layout); hold(ax,'on');
for c=1:toy.n_channels
 plot(ax,1:toy.n_epochs,1000*metadata.artifact_onsets_s(:,c),'.','Color',grey,'HandleVisibility','off');
end
plot(ax,1:toy.n_epochs,1000*mean(metadata.artifact_onsets_s,2),'-o','Color',blue,'DisplayName','Mean across channels');
yline(ax,3,':','Color',[.4 .4 .4],'DisplayName','Nominal onset: 3 ms');
xlabel(ax,'Stimulus / epoch'); ylabel(ax,'Residual artifact onset (ms)');
title(ax,'Stimulus jitter + channel latency');
jitter_legend=legend(ax,'Orientation','horizontal','FontSize',9,'Box','off');
jitter_legend.Layout.Tile='south';

%% 5. Continue the host pipeline
% Replace data/times with real preprocessed epochs; keep matching good EEG
% channels in MATLAB and Python because screening depends on that population.
% Gaussian filling remains enabled; inspect diagnostics, then continue ICA.
data_for_ica=corrected;
