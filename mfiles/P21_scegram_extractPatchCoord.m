
% BASE_PATH = '';  % Set your path here
% coords = P21_scegram_extractPatchCoord([BASE_PATH '\4a_scegramSubset_QualtricsOutput_raw\patchesNamesQualCodesLinks.xls']);
% save('P21_scegram_originalPatchCoords.mat', 'coords')



function [uniqueData] = extractPatchCoord(filename)
    % EXTRACTPATCHCOORD Extracts unique combinations of radius, yc, and xc from patch names
    %   This function reads an Excel file containing patch names and extracts the
    %   embedded yc, xc, and radius values, returning only unique combinations
    %
    % Input:
    %   filename - String, path to the Excel file
    %
    % Output:
    %   uniqueData - Table with columns: radius, yc, xc
    %
    % Example:
    %   data = extractPatchCoord('patchesNamesQualCodesLinks.xls');

    % Read the Excel file
    [~, ~, raw] = xlsread(filename);
    
    % Get the patch names from the first column (excluding header)
    patchNames = raw(2:end, 1);
    
    % Initialize arrays to store extracted values
    numPatches = length(patchNames);
    yc = zeros(numPatches, 1);
    xc = zeros(numPatches, 1);
    radius = zeros(numPatches, 1);
    
    % Extract values using regular expressions
    for i = 1:numPatches
        patchName = patchNames{i};
        
        % Extract yc value
        ycMatch = regexp(patchName, 'yc_(\d+)', 'tokens');
        if ~isempty(ycMatch)
            yc(i) = str2double(ycMatch{1}{1});
        end
        
        % Extract xc value
        xcMatch = regexp(patchName, 'xc_(\d+)', 'tokens');
        if ~isempty(xcMatch)
            xc(i) = str2double(xcMatch{1}{1});
        end
        
        % Extract radius value
        radMatch = regexp(patchName, 'rad_(\d+)', 'tokens');
        if ~isempty(radMatch)
            radius(i) = str2double(radMatch{1}{1});
        end
    end
    
    % Create a matrix of all values
    allData = [radius, yc, xc];
    
    % Find unique combinations
    uniqueCombinations = unique(allData, 'rows');
    
    % Convert to table with appropriate column names
    uniqueData = array2table(uniqueCombinations, 'VariableNames', {'radius', 'yc', 'xc'});
end