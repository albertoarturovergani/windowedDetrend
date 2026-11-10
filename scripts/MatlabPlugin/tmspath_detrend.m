function [cleaned,trend,info] = tmspath_detrend(data,times,cfg)
% Standalone post-epoching detrending, no toolbox required.
% data: epochs x channels x samples. times: seconds. channels: 1-based.
% Empty channels means all. No random replacement or filtering.
if nargin<3, cfg=tmspath_config(); end
def=tmspath_config(); names=fieldnames(cfg);
assert(all(ismember(names,fieldnames(def))),'Unknown configuration field');
for i=1:numel(names), def.(names{i})=cfg.(names{i}); end
cfg=def; data=double(data); t=double(times(:));
assert(isvector(times) && ndims(data)<=3 && size(data,3)==numel(t) && ~isempty(data),'Expected epochs x channels x samples');
assert(all(isfinite(data(:))) && all(isfinite(t)) && all(diff(t)>0),'Finite input and increasing times required');
finish=cfg.end_s; if isempty(finish), finish=t(end); end
assert(t(1)<=cfg.onset_s && cfg.onset_s<finish && finish<=t(end)+1e-12,'Invalid correction interval');
assert(ismember(cfg.mode,{'windowed','global'}),'Invalid mode');
assert(cfg.tau_min_s>0 && cfg.tau_max_s>cfg.tau_min_s && cfg.grid_size>=3 && cfg.refinements>=0,'Invalid search parameters');
assert(cfg.grid_size==fix(cfg.grid_size) && cfg.refinements==fix(cfg.refinements),'Grid counts must be integers');
for d=[cfg.rise_degree cfg.decay_degree cfg.pre_degree], assert(d>=0 && d==fix(d),'Invalid degree'); end
channels=cfg.channels; if isempty(channels), channels=1:size(data,2); end
assert(numel(unique(channels))==numel(channels) && all(channels>=1 & channels<=size(data,2) & channels==fix(channels)),'Invalid channels');
trend=zeros(size(data)); records={};
for e=1:size(data,1)
 for c=channels(:)'
  y=reshape(data(e,c,:),[],1); z=zeros(size(t)); rec=struct('epoch',e,'channel',c);
  if ~isempty(cfg.pre_degree)
   mask=t<cfg.onset_s;
   [z(mask),rec.pre]=fit_segment(t(mask)-t(find(mask,1)),y(mask),'poly',cfg.pre_degree,cfg,[]);
  end
  anchor=[]; if cfg.constrain_rise, anchor=0; end
  if strcmp(cfg.mode,'global')
   mask=t>=cfg.onset_s & t<=finish;
   [z(mask),rec.global]=fit_segment(t(mask)-cfg.onset_s,y(mask),cfg.global_model,cfg.decay_degree,cfg,anchor);
  else
   peak=cfg.peak_s;
   if isempty(peak)
    ids=find(t>=cfg.onset_s & t<=min(cfg.peak_search_end_s,finish));
    assert(~isempty(ids),'Empty peak search interval');
    [~,k]=max(abs(y(ids))); peak=t(ids(k));
   end
   assert(cfg.onset_s<peak && peak<finish,'Peak must lie inside correction interval');
   rise=t>=cfg.onset_s & t<peak; decay=t>=peak & t<=finish;
   [z(rise),rec.rise]=fit_segment(t(rise)-cfg.onset_s,y(rise),cfg.rise_model,cfg.rise_degree,cfg,anchor);
   anchor=[]; if cfg.constrain_decay, anchor=interp1(t,y,peak); end
   [z(decay),rec.decay]=fit_segment(t(decay)-peak,y(decay),cfg.decay_model,cfg.decay_degree,cfg,anchor);
   rec.peak_s=peak;
  end
  trend(e,c,:)=reshape(z,1,1,[]); records{end+1}=rec; %#ok<AGROW>
 end
end
cleaned=data-trend; info=struct('config',cfg,'fits',{records});
end

function [fit,info]=fit_segment(x,y,model,degree,cfg,anchor)
if strcmp(model,'lagrange')
 assert(numel(x)>=2 && all(diff(x)>0),'Lagrange requires at least two distinct increasing nodes');
 assert(isempty(anchor),'Lagrange uses observed nodes; disable the start constraint');
 % The Lagrange interpolant equals the observed signal at every input node.
 % No high-degree Vandermonde solve is needed for evaluation at those nodes.
 fit=y;
 info=struct('model',model,'degree',numel(x)-1,'node_times_s',x, ...
  'node_values',y,'tau_s',[],'coefficients',[],'residual_mse',0, ...
  'anchor',[],'scale_s',max(max(x),1e-12),'tau_at_bound',false);
 return;
end
nmin=4; if strcmp(model,'poly'), nmin=max(nmin,degree+2); end
assert(numel(x)>=nmin,'Too few samples for model/degree');
switch model
 case 'poly', ntheta=0;
 case {'singlerise','singledecay','alpha'}, ntheta=1;
 case {'biexp','doubledecay'}, ntheta=2;
 otherwise, error('Unsupported model: %s',model);
end
scale=max(max(x),1e-12); bounds=log([cfg.tau_min_s cfg.tau_max_s]);
lo=repmat(bounds(1),1,ntheta); hi=repmat(bounds(2),1,ntheta);
best=Inf; bestTheta=[]; bestCoef=[]; fit=[];
levels=cfg.refinements; if ntheta==0, levels=0; end
for level=0:levels
 n=cfg.grid_size; if level>0, n=7; end
 if ntheta==0
  candidates=zeros(1,0);
 elseif ntheta==1
  candidates=exp(linspace(lo(1),hi(1),n))';
 else
  a=exp(linspace(lo(1),hi(1),n)); b=exp(linspace(lo(2),hi(2),n));
  candidates=[repelem(a(:),n) repmat(b(:),n,1)];
 end
 for k=1:size(candidates,1)
  theta=candidates(k,:);
  if strcmp(model,'doubledecay') && theta(1)>=theta(2), continue; end
  B=basis(x,model,theta,degree,scale); b0=basis(0,model,theta,degree,scale);
  if ~strcmp(model,'poly') && cfg.fit_offset, B=[B ones(size(x))]; b0=[b0 1]; end
  coef=zeros(size(B,2),1); keep=true(1,size(B,2)); pivot=false;
  if ~isempty(anchor)
   [v,j]=max(abs(b0));
   if v>1e-12
    pivot=true; keep(j)=false; coef(j)=anchor/b0(j);
    if any(keep), X=B(:,keep)-B(:,j)*reshape(b0(keep)/b0(j),1,[]); else, X=zeros(numel(x),0); end
    rhs=y-B*coef;
   else
    assert(abs(anchor)<=1e-12,'Incompatible start constraint'); X=B; rhs=y;
   end
  else
   X=B; rhs=y;
  end
  if ~isempty(X)
   if rank(X)<size(X,2), continue; end
   coef(keep)=X\rhs;
   if pivot, coef(j)=(anchor-b0(keep)*coef(keep))/b0(j); end
  end
  prediction=B*coef; score=mean((y-prediction).^2);
  if score<best, best=score; fit=prediction; bestTheta=theta; bestCoef=coef; end
 end
 assert(isfinite(best),'No full-rank fit');
 if ntheta>0
  step=(hi-lo)/(n-1); center=log(bestTheta);
  lo=max(bounds(1),center-step); hi=min(bounds(2),center+step);
 end
end
info=struct('model',model,'tau_s',bestTheta,'coefficients',bestCoef, ...
 'residual_mse',best,'anchor',anchor,'scale_s',scale, ...
 'tau_at_bound',any(abs(bestTheta-cfg.tau_min_s)<1e-8+1e-5*cfg.tau_min_s) || any(abs(bestTheta-cfg.tau_max_s)<1e-8+1e-5*cfg.tau_max_s));
end

function B=basis(x,model,theta,degree,scale)
x=x(:);
switch model
 case 'poly', B=(x/scale).^(0:degree);
 case 'singlerise', B=-expm1(-x/theta(1));
 case 'singledecay', B=exp(-x/theta(1));
 case 'biexp', B=-expm1(-x/theta(1)).*exp(-x/theta(2));
 case 'doubledecay', B=[exp(-x/theta(1)) exp(-x/theta(2))];
 case 'alpha', B=x/theta(1).*exp(-x/theta(1));
end
end
