close all
clear
clc

export_dir = "exports";

model_names = ["cnn2", "cnn3", "fnn2", "fnn3"];
seeds = 0:9;

all_results = struct();

for m = 1:numel(model_names)

    model_name = model_names(m);

    fprintf("\n========================================\n");
    fprintf("Evaluating model: %s\n", model_name);
    fprintf("========================================\n");

    model_results = struct();

    for s = 1:numel(seeds)

        seed = seeds(s);

        weight_file = fullfile( ...
            export_dir, ...
            sprintf("%s_seed%d_weights.mat", model_name, seed) ...
        );

        fprintf("Seed %d: %s\n", seed, weight_file);

        % -------------------------------------------------------------
        % Run your existing single-model Lipschitz evaluation here.
        %
        % This function should load the weights and return a struct
        % containing variables such as Lip1, Lip2, LipTotal, etc.
        % -------------------------------------------------------------
        lip_values = evaluate_single_model_lipschitz(weight_file, model_name);

        lip_names = fieldnames(lip_values);

        for k = 1:numel(lip_names)

            lip_name = lip_names{k};

            if startsWith(lip_name, "Lip")

                value = lip_values.(lip_name);

                if ~isfield(model_results, lip_name)
                    model_results.(lip_name) = [];
                end

                model_results.(lip_name)(s, :) = value;
            end
        end
    end

    % -------------------------------------------------------------
    % Compute mean and standard deviation for every Lip* variable
    % -------------------------------------------------------------
    summary = struct();
    lip_names = fieldnames(model_results);

    for k = 1:numel(lip_names)

        lip_name = lip_names{k};

        values = model_results.(lip_name);

        summary.(lip_name).values = values;
        summary.(lip_name).mean = mean(values, 1);
        summary.(lip_name).std = std(values, 0, 1);

        fprintf("%s.%s:\n", model_name, lip_name);
        fprintf("  mean = ");
        disp(summary.(lip_name).mean);
        fprintf("  std  = ");
        disp(summary.(lip_name).std);
    end

    all_results.(model_name).individual = model_results;
    all_results.(model_name).summary = summary;
end

save(fullfile(export_dir, "lipschitz_summary_all_models.mat"), "all_results");

fprintf("\nSaved Lipschitz summary to:\n");
fprintf("%s\n", fullfile(export_dir, "lipschitz_summary_all_models.mat"));