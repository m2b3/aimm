clear all
% Wrapper around the AWS saliency model of Garcia-Diaz et al. (2012). The model
% itself is not distributed here - obtain aws() from the authors:
%   http://persoal.citius.usc.es/xose.vidal/research/aws/AWSmodel.html
%
% Run this from the repository root - every path below is relative to it.

% Dataset selection - specify which dataset to process
DATASET = 'P21_scegram';  % 'P21_scegram', 'HH25_indoor', 'HH25_outdoor', 'TEST'

inpath = ['data\' DATASET '\images\'];
outpath = ['data\' DATASET '\maps\aws_ncb\'];

% Create output directory if it doesn't exist
if ~exist(outpath, 'dir')
    mkdir(outpath);
end

% Get all image files
files = dir([inpath '*.png']);

% Process each image
for i = 1 :length(files)
    
    % Load image
    img = imread(fullfile(inpath, files(i).name));
    
    % Process with gbvs
    map = aws(img);
    map_norm = (map - min(map(:))) / (max(map(:)) - min(map(:)));

    % Save as PNG with same filename
    [~, name, ~] = fileparts(files(i).name);
    imwrite(map_norm, fullfile(outpath, [name '.png']));
     
%     % Save as MAT with same filename
%     save(fullfile(outpath, [name '.mat']), 'map');

end

