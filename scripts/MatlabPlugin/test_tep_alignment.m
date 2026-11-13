function test_tep_alignment()
t=(-100:200)'/1000;u=max(t,0);a=-min(u/.01,1).*exp(-max(u-.01,0)/.05);
cfg=tmspath_config();assert(cfg.pre_degree==2 && ~cfg.constrain_decay);
assert(cfg.onset_s==0 && cfg.peak_search_end_s==.020 && cfg.overall_degree==1);
cfg.channel_policy='all';cfg.peak_s=.010;cfg.fill_mode='none';
y=reshape(2+3*t+4*t.^2,1,1,[]);[cleaned,~,info]=tmspath_detrend(y,t,cfg);
assert(max(abs(cleaned(:)))<3e-14);r=info.fits{1};
assert(r.windows.pre.last==r.windows.rise.first && r.windows.decay.first==r.windows.rise.last+1);
cfg=tmspath_config();cfg.fill_mode='none';
data=reshape([40;2;2;2;2;2]*a',1,6,[]);
[~,~,info]=tmspath_detrend(data,t,cfg);
assert(isequal(info.channel_selection.offset_channels,1));
for k=1:6, assert(info.fits{k}.decay.degree==3-(k>1)); end
data=zeros(2,3,numel(t));data(1,:,:)=repmat(reshape(a,1,1,[]),1,3,1);data(2,:,:)=-data(1,:,:);
[~,~,info]=tmspath_detrend(data,t,cfg);assert(strcmp(info.operation,'overall'));
assert(isempty(info.channel_selection.offset_channels));
cfg.overall_degree=[];cleaned=tmspath_detrend(data,t,cfg);assert(isequal(cleaned,data));
fprintf('MATLAB published TEP defaults, masks, screening and fallback tests passed.\n');
end
