%% export_lipschitz_results_to_excel.m

clear; clc;

% Ordner mit den .mat-Dateien
dataFolder = "results";   % ggf. anpassen, z.B. "results"

% Ausgabedatei
outFile = fullfile(dataFolder, "mnist_lipschitz_results.xlsx");

% Dateinamen ohne "_instanceX.mat"
modelFiles = {
    "mnist_LeNet5"
    "mnist_2C2F"
    "mnist_resnet_fc"
    "mnist_resnet"
};

% Namen für die Tabelle
modelNames = {
    "LeNet-5"
    "2C2F"
    "FC-R18"
    "C-R18"
};

% Variablen, die aus jeder .mat-Datei gelesen werden sollen
varNames = {
    "Lip_GLipSDP"
    "time_GLipSDP"
    "Lip_MP"
    "Lip_S_GLipSDP"
    "time_S_GLipSDP"
    "Lip_S_LipSDP"
    "time_S_LipSDP"
    "Lip_GLipSDP_l"
    "time_GLipSDP_l"
};

% Tabelle initialisieren
rows = {};

for m = 1:numel(modelFiles)
    for inst = 1:5

        filename = sprintf("%s_instance%d.mat", modelFiles{m}, inst);
        filepath = fullfile(dataFolder, filename);

        % Basisinformationen
        row = cell(1, 2 + numel(varNames));
        row{1} = modelNames{m};
        row{2} = inst;

        if ~isfile(filepath)
            warning("Datei nicht gefunden: %s", filepath);
            row(3:end) = {NaN};
        else
            S = load(filepath);

            for j = 1:numel(varNames)
                v = varNames{j};

                if isfield(S, v)
                    value = S.(v);

                    % Falls Skalar, direkt speichern
                    if isnumeric(value) && isscalar(value)
                        row{2+j} = double(value);

                    % Falls numerisch, aber kein Skalar: erstes Element nehmen
                    elseif isnumeric(value) && ~isempty(value)
                        row{2+j} = double(value(1));

                    else
                        row{2+j} = NaN;
                        warning("Variable %s in %s ist nicht numerisch.", v, filename);
                    end
                else
                    row{2+j} = NaN;
                    warning("Variable %s fehlt in %s.", v, filename);
                end
            end
        end

        rows(end+1, :) = row; %#ok<SAGROW>
    end
end

% Spaltennamen
columnNames = ["Model", "Instance", varNames{:}];

% Cell array in Tabelle umwandeln
T = cell2table(rows, "VariableNames", columnNames);

% Excel-Datei schreiben
writetable(T, outFile);

fprintf("Excel-Tabelle geschrieben nach:\n%s\n", outFile);