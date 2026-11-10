function [EEGout,trend,info]=tmspath_eeglab(EEG,cfg)
% EEGLAB data: channels x samples x trials; EEG.times is milliseconds.
% Set cfg.channels explicitly to avoid correcting EOG/other non-EEG channels.
if nargin<2, cfg=tmspath_config(); end
assert(isnumeric(EEG.data),'Load epoched EEG.data into memory first');
assert(EEG.trials>=1 && EEG.pnts==numel(EEG.times),'Invalid epoched EEG');
[cleaned,z,info]=tmspath_detrend(permute(double(EEG.data),[3 1 2]),double(EEG.times)/1000,cfg);
EEGout=EEG; EEGout.data=permute(cleaned,[2 3 1]); trend=permute(z,[2 3 1]);
end
