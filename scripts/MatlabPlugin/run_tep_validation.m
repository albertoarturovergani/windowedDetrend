function report = run_tep_validation(input_file,output_file)
% Compare native MATLAB with Python and the unchanged published TEP functions.
root=fileparts(mfilename('fullpath'));
if nargin<1, input_file=fullfile(root,'..','results','tep_alignment','shared_fixture.mat'); end
if nargin<2, output_file=fullfile(fileparts(input_file),'matlab_report.json'); end
fixture=load(input_file);max_trend=0;max_output=0;max_pipeline=0;max_parameters=0;max_slopes=0;max_z=0;
traces=0;mask_checks=0;gaussian_traces=0;
for i=1:numel(fixture.cases)
 c=fixture.cases{i};[cleaned,trend,info]=tmspath_detrend(c.data,c.times,c.cfg);
 assert(strcmp(info.operation,c.operation),'Operation differs');
 assert(isequal(info.channel_selection.offset_channels(:),double(c.offset_channels(:))),'Offset-channel selection differs');
 max_trend=max(max_trend,max(abs(trend(:)-c.python_trend(:))));
 if ~isempty(c.slopes)
  max_slopes=max(max_slopes,max(abs(info.channel_selection.slopes(:)-c.slopes(:))));
  max_z=max(max_z,max(abs(info.channel_selection.mean_z(:)-c.mean_z(:))));
 end
 protected=true(size(cleaned));
 for k=1:numel(info.fits)
  r=info.fits{k};traces=traces+1;
  if ~isfield(r,'rise'), continue; end
  row=find(c.degree_rows(:,1)==r.epoch & c.degree_rows(:,2)==r.channel);
  assert(r.decay.degree==c.degree_rows(row,3),'Per-channel decay degree differs');
  w=r.windows;
  bounds=[r.epoch r.channel w.pre.first w.pre.last w.rise.first w.rise.last w.decay.first w.decay.last];
  assert(isequal(bounds,double(c.mask_bounds(row,:))),'Original pipeline masks differ');mask_checks=mask_checks+1;
  if isfield(r,'fill')
   f=r.fill;fr=find(c.fill_bounds(:,1)==r.epoch & c.fill_bounds(:,2)==r.channel);
   bounds=[r.epoch r.channel f.source_first f.source_last f.target_first f.target_last double(f.source_fallback)];
   assert(isequal(bounds,double(c.fill_bounds(fr,:))),'Fill masks differ');
   if strcmp(f.mode,'gaussian')
    protected(r.epoch,r.channel,f.target_first:f.target_last)=false;gaussian_traces=gaussian_traces+1;
    max_parameters=max(max_parameters,max(abs([f.mean f.std]-c.noise_parameters(fr,:))));
   end
  end
 end
 max_output=max(max_output,max(abs(cleaned(protected)-c.python_cleaned(protected))));
 pipeline_mask=protected & logical(c.pipeline_compare_mask);
 max_pipeline=max(max_pipeline,max(abs(cleaned(pipeline_mask)-c.pipeline_cleaned(pipeline_mask))));
end
assert(max_trend<1e-9 && max_output<1e-9 && max_pipeline<2e-8 && max_parameters<1e-12 && max_slopes<1e-9 && max_z<1e-10,'TEP parity failed');
report=struct('passed',true,'cases',numel(fixture.cases),'processed_traces',traces, ...
 'original_mask_checks',mask_checks,'gaussian_traces',gaussian_traces, ...
 'max_python_matlab_trend_difference',max_trend,'max_deterministic_output_difference',max_output, ...
 'max_pipeline_difference_on_comparable_samples',max_pipeline,'max_gaussian_parameter_difference',max_parameters, ...
 'max_slope_difference',max_slopes,'max_mean_z_difference',max_z,'matlab_version',version, ...
 'gaussian_comparison','Native random samples differ; original masks and distribution parameters agree.', ...
 'pipeline_comparison','Compare all samples except native Gaussian draws and original unfilled Lagrange numerical residuals.');
fid=fopen(output_file,'w');assert(fid>=0);closer=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(report));disp(report);
end
