% P21_scegram_stitch_meaning_maps.m
%
% Constructs unsmoothed meaning maps from rated image patches.
% Dataset: P21_scegram, SCEGRAM, CON images only.
%
% Pipeline:
%   1. For each image, compute the per-pixel mean rating across all fine
%      patches covering that pixel   -> fineMap
%   2. Do the same for coarse patches                                    -> coarseMap
%   3. Average the two maps                                              -> meaningMap
%
% Output (one pair of files per image, named after the scene):
%   <sceneId>.mat  - contains meaningMap, fineMap, coarseMap (raw float values)
%   <sceneId>.png  - normalized grayscale (0 = min, 1 = max meaning)
%
% Patch geometry is hardcoded from the original metaData_fine.txt /
% metaData_coarse.txt files. No smoothing or interpolation is applied.

clear all; close all; clc;

%% ---- Paths ---------------------------------------------------------------

csvPath   = ['\4a1_scegramSubset_processedRatings_SUR_FIN_18-Jun-2019 14_18_33\' ...
             'subsetScegram_ratingsTabAgg_18-Jun-2019 14_18_33.csv'];

outputDir = '';

%% ---- Patch geometry (from metaData_fine.txt and metaData_coarse.txt) ----
%
% Note: "fine" = smaller patches (radius 53), "coarse" = larger (radius 123)
%       targetImHeight/Width is the UNPADDED image size (524 x 688).
%       Patch center coordinates in the CSV are expressed on the PADDED canvas.

fine.patchRad  = 53;
fine.padUp     = 52;
fine.padDown   = 54;
fine.padRight  = 30;
fine.padLeft   = 28;
fine.targetH   = 524;
fine.targetW   = 688;

coarse.patchRad = 123;
coarse.padUp    = 103;
coarse.padDown  = 106;
coarse.padRight = 121;
coarse.padLeft  = 118;
coarse.targetH  = 524;
coarse.targetW  = 688;

%% ---- Setup ---------------------------------------------------------------

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created output directory: %s\n', outputDir);
end

%% ---- Load ratings --------------------------------------------------------

T = readtable(csvPath);

% Keep only CON images
T = T(contains(T.patchName, '_CON'), :);

% Per-patch mean rating (average over the 3 raters)
T.meanRating = (T.rating1 + T.rating2 + T.rating3) / 3;

%% ---- Parse patch names ---------------------------------------------------
% Format: resScene11_CON_yc_124_xc_124_rad_123_code_qco004841

nRows    = height(T);
sceneIds = cell(nRows, 1);
yCenters = zeros(nRows, 1);
xCenters = zeros(nRows, 1);

for i = 1:nRows
    name = T.patchName{i};
    conIdx       = strfind(name, '_CON');
    sceneIds{i}  = name(1 : conIdx+3);            % e.g. 'resScene11_CON'
    ycTok        = regexp(name, 'yc_(\d+)', 'tokens');
    xcTok        = regexp(name, 'xc_(\d+)', 'tokens');
    yCenters(i)  = str2double(ycTok{1}{1});
    xCenters(i)  = str2double(xcTok{1}{1});
end

T.sceneId = sceneIds;
T.yc      = yCenters;
T.xc      = xCenters;

%% ---- Process each scene --------------------------------------------------

uniqueScenes = unique(T.sceneId);
nScenes      = numel(uniqueScenes);
fprintf('Found %d CON scenes\n\n', nScenes);

for s = 1:nScenes
    sceneId = uniqueScenes{s};
    fprintf('Processing %s ...\n', sceneId);

    sceneT  = T(strcmp(T.sceneId, sceneId), :);
    fineT   = sceneT(logical(sceneT.isFine),   :);
    coarseT = sceneT(logical(sceneT.isCoarse), :);

    fineMap   = buildMeaningMap(fineT,   fine);
    coarseMap = buildMeaningMap(coarseT, coarse);

    % Average the two scales
    meaningMap = (fineMap + coarseMap) / 2;

    % Save .mat (raw floating-point values)
    save(fullfile(outputDir, [sceneId, '.mat']), ...
         'meaningMap', 'fineMap', 'coarseMap');

    % Save .png (normalized grayscale: 0 = min, 1 = max)
    mapNorm = normalizeMap(meaningMap);
    imwrite(mapNorm, fullfile(outputDir, [sceneId, '.png']));

    fprintf('  -> saved %s\n', sceneId);
end

fprintf('\nDone! %d images written to:\n  %s\n', nScenes, outputDir);


%% =========================================================================
%  Helper functions
%% =========================================================================

function map = buildMeaningMap(patchTab, geo)
% Build a per-pixel mean rating map for one patch scale.
%
% For every pixel, the value is the mean rating across all patches whose
% circular footprint covers that pixel.  Maps are built on the padded canvas
% and then unpadded to the original image dimensions.

    patchRad = geo.patchRad;
    paddedH  = geo.targetH + geo.padUp  + geo.padDown;
    paddedW  = geo.targetW + geo.padLeft + geo.padRight;

    circMask = makeCircleMask(patchRad);   % 1 inside circle, 0 outside

    ratingAcc = zeros(paddedH, paddedW);
    countAcc  = zeros(paddedH, paddedW);

    for i = 1:height(patchTab)
        yc     = patchTab.yc(i);
        xc     = patchTab.xc(i);
        sumRat = patchTab.rating1(i) + patchTab.rating2(i) + patchTab.rating3(i);

        rows = (yc - patchRad) : (yc + patchRad);
        cols = (xc - patchRad) : (xc + patchRad);

        ratingAcc(rows, cols) = ratingAcc(rows, cols) + sumRat .* circMask;
        countAcc(rows, cols)  = countAcc(rows, cols)  + 3      .* circMask;
    end

    % Divide accumulated ratings by total rater-counts
    paddedMap           = zeros(paddedH, paddedW);
    validMask           = countAcc > 0;
    paddedMap(validMask) = ratingAcc(validMask) ./ countAcc(validMask);

    % Remove padding to return to original image dimensions
    map = unpadd(paddedMap, geo.padUp, geo.padDown, geo.padRight, geo.padLeft);
end


function mask = makeCircleMask(radius)
% Binary circular mask (1 = inside circle, 0 = outside).
% Side length is 2*radius+1 so the center pixel aligns exactly.
    side        = 2 * radius + 1;
    [xx, yy]    = meshgrid(1:side, 1:side);
    cx          = radius + 1;
    cy          = radius + 1;
    mask        = double(((xx - cx).^2 + (yy - cy).^2) <= radius^2);
end


function outIm = unpadd(inIm, padU, padD, padR, padL)
% Remove padding from all four edges.
% Replicates the logic of Unpadd4edges.m from the original codebase.
    outIm = inIm((padU+1):end, :);
    outIm = outIm(1:(end-padD), :);
    outIm = rot90(outIm);           % rotate CCW so left/right become top/bottom
    outIm = outIm((padR+1):end, :);
    outIm = outIm(1:(end-padL), :);
    outIm = rot90(outIm, 3);        % rotate CW back to original orientation
end


function normMap = normalizeMap(inMap)
% Scale map to [0, 1] range.
    lo      = min(inMap(:));
    hi      = max(inMap(:));
    normMap = (inMap - lo) / (hi - lo);
end
