function test_plugin()
t=(0:2499)'/10000; [gt,a]=tmspath_synthetic(t,.003,.2);
cfg=tmspath_config();cfg.fill_mode='none';cfg.onset_s=.010;cfg.peak_s=.015;
assert(strcmp(cfg.rise_model,'lagrange') && strcmp(cfg.decay_model,'poly') && cfg.decay_degree==3);
[defaultZ,defaultTrend,defaultInfo]=tmspath_detrend(reshape(gt+a,1,1,[]),t,cfg);
rise=t>=.010 & t<.015; decay=t>=.015;
assert(all(defaultZ(1,1,rise)==0));
assert(defaultInfo.fits{1}.rise.degree==sum(rise)-1);
assert(numel(defaultInfo.fits{1}.decay.coefficients)==4);
assert(max(abs(diff(reshape(defaultTrend(1,1,decay),[],1),4)))<1e-10);
cfg.rise_model='singlerise';cfg.decay_model='singledecay';cfg.constrain_rise=true;
x=zeros(2,2,numel(t));x(:,1,:)=repmat(reshape(gt+a,1,1,[]),2,1,1);x(:,2,:)=17;
cfg.channels=1; [clean,trend]=tmspath_detrend(x,t,cfg);
assert(isequal(size(clean),size(x)) && isequal(clean(:,2,:),x(:,2,:)));
assert(max(abs(clean(:)+trend(:)-x(:)))<1e-12);
assert(isequal(clean(:,:,t<cfg.onset_s),x(:,:,t<cfg.onset_s)));
assert(mean((reshape(clean(1,1,:),[],1)-gt).^2)<mean(a.^2));
% Exact windowed artifact at a grid tau should be removed to round-off.
cfg.tau_min_s=.003;cfg.tau_max_s=.2;cfg.grid_size=21;
z=tmspath_detrend(reshape(a,1,1,[]),t,cfg);assert(max(abs(z(:)))<1e-9);
% Invalid times must fail.
failed=false;try, tmspath_detrend(x,flipud(t),cfg);catch, failed=true;end
assert(failed);
% EEGLAB axis/unit adapter preserves metadata and unselected channels.
EEG=struct('data',permute(x,[2 3 1]),'times',t'*1000,'trials',2,'pnts',numel(t),'setname','fixture');
out=tmspath_eeglab(EEG,cfg);assert(strcmp(out.setname,'fixture'));assert(isequal(out.data(2,:,:),EEG.data(2,:,:)));
fprintf('MATLAB core and EEGLAB tests passed.\n');
end
