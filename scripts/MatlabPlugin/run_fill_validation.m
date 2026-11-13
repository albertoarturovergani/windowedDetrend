function report = run_fill_validation(input_file,output_file)
% Run native MATLAB on fixtures from PythonPlugin/validate_filling.py.
root=fileparts(mfilename('fullpath'));
if nargin<1, input_file=fullfile(root,'..','results','filling_validation','shared_fixture.mat'); end
if nargin<2, output_file=fullfile(fileparts(input_file),'matlab_report.json'); end
fixture=load(input_file); max_output=0; max_trend=0; max_pipeline=0; max_fft=0; traces=0;
max_noise_parameters=0; max_outside_fill=0; gaussian_traces=0; resample_traces=0;
for i=1:numel(fixture.cases)
 c=fixture.cases{i}; [cleaned,trend,info]=tmspath_detrend(c.data,c.times,c.cfg);
 is_gaussian=strcmp(c.cfg.fill_mode,'gaussian');
 if ~is_gaussian
  max_output=max(max_output,max(abs(cleaned(:)-c.python_cleaned(:))));
  max_pipeline=max(max_pipeline,max(abs(cleaned(:)-c.pipeline_cleaned(:))));
 end
 max_trend=max(max_trend,max(abs(trend(:)-c.python_trend(:))));
 protected=true(size(cleaned));
 for k=1:numel(info.fits)
  r=info.fits{k}; f=r.fill;
  bounds=[r.epoch r.channel f.source_first f.source_last f.target_first f.target_last double(f.source_fallback)];
  assert(isequal(bounds,double(c.fill_bounds(k,:))),'Source or replacement boundaries differ');
  protected(r.epoch,r.channel,f.target_first:f.target_last)=false;
  if is_gaussian
   max_noise_parameters=max(max_noise_parameters,max(abs([f.mean f.std]-c.noise_parameters(k,:))));
   expected=reshape(cleaned(r.epoch,r.channel,f.target_first:f.target_last),[],1);
   assert(isequal(expected,f.values)); gaussian_traces=gaussian_traces+1;
  else
   resample_traces=resample_traces+1;
  end
  traces=traces+1;
 end
 if any(protected(:))
  max_outside_fill=max(max_outside_fill,max(abs(cleaned(protected)-c.python_cleaned(protected))));
 end
end
for i=1:numel(fixture.fft_cases)
 c=fixture.fft_cases{i}; actual=tmspath_resample_fft(c.values,c.count);
 max_fft=max(max_fft,max(abs(actual(:)-c.expected(:))));
end
assert(max_output<1e-9 && max_trend<1e-9 && max_pipeline<1e-9 && max_fft<1e-12 && max_noise_parameters<1e-12 && max_outside_fill<1e-9,'Filling parity failed');
report=struct('passed',true,'cases',numel(fixture.cases),'processed_traces',traces, ...
 'gaussian_traces',gaussian_traces,'resample_traces',resample_traces, ...
 'fourier_cases',numel(fixture.fft_cases),'max_resample_python_matlab_output_difference',max_output, ...
 'max_python_matlab_trend_difference',max_trend,'max_original_pipeline_difference',max_pipeline, ...
 'max_scipy_fourier_difference',max_fft,'max_gaussian_distribution_parameter_difference',max_noise_parameters, ...
 'max_difference_outside_replacement',max_outside_fill,'matlab_version',version, ...
 'gaussian_comparison','Native RNG draws differ. Compare fitted trend, source/target masks, mean/SD, seeded repeatability, and Gaussian distribution tests.');
fid=fopen(output_file,'w'); assert(fid>=0,'Cannot open report');
closer=onCleanup(@()fclose(fid)); fprintf(fid,'%s\n',jsonencode(report));
disp(report);
end
