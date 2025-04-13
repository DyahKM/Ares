% Parameter Tuning for H-FUSION and ARES with Monthly Pollution Variables
% This script uses monthly pollution variables as references for validation

%% 1. Load and prepare data
addpath('HFusion')
addpath('AresFilter')
clc; clear;

% Load your aggregated report data (using the same format as in your example)
reports = [
     1,  31, 43787;   % Jan 2020
    32,  60, 40427;   % Feb 2020 (29 days, leap year)
    61,  91, 39398;   % Mar 2020
    92, 121, 25369;   % Apr 2020
   122, 152, 19049;   % May 2020
   153, 182, 26593;   % Jun 2020
   183, 213, 26878;   % Jul 2020
   214, 244, 23179;   % Aug 2020
   245, 274, 24317;   % Sep 2020
   275, 305, 20644;   % Oct 2020
   306, 335, 21124;   % Nov 2020
   336, 366, 16747;   % Dec 2020
   367, 397, 35053;   % Jan 2021
   398, 425, 31532;   % Feb 2021 (28 days)
   426, 456, 39848;   % Mar 2021
   457, 486, 38948;   % Apr 2021
   487, 517, 34749;   % May 2021
   518, 547, 37652;   % Jun 2021
   548, 578, 25075;   % Jul 2021
   579, 609, 30670;   % Aug 2021
   610, 639, 39631;   % Sep 2021
   640, 670, 42452;   % Oct 2021
   671, 700, 45619;   % Nov 2021
   701, 731, 44017;   % Dec 2021
   732, 762, 66025;   % Jan 2022
   763, 789, 48431;   % Feb 2022 (28 days)
   790, 820, 56857;   % Mar 2022
   821, 850, 52675;   % Apr 2022
   851, 881, 54903;   % May 2022
   882, 911, 66734;   % Jun 2022
   912, 942, 63507;   % Jul 2022
   943, 973, 67759;   % Aug 2022
   974,1003, 69874;   % Sep 2022
  1004,1034, 69597;   % Oct 2022
  1035,1064, 68806;   % Nov 2022
  1065,1095, 56626;   % Dec 2022
  1096,1126, 83897;   % Jan 2023
  1127,1154, 73983;   % Feb 2023 (28 days)
  1155,1185, 77314;   % Mar 2023
  1186,1215, 56881;   % Apr 2023
  1216,1246, 80216;   % May 2023
  1247,1276, 69611;   % Jun 2023
  1277,1307, 78302;   % Jul 2023
  1308,1338, 85188;   % Aug 2023
  1339,1368, 78406;   % Sep 2023
  1369,1399, 83093;   % Oct 2023
  1400,1429, 77180;   % Nov 2023
  1430,1460, 63072;   % Dec 2023
  1461,1491, 75589;   % Jan 2024
  1492,1519, 65536;   % Feb 2024 (29 days, leap year)
  1520,1550, 68438;   % Mar 2024
  1551,1580, 66651;   % Apr 2024
  1581,1611, 76249;   % May 2024
  1612,1641, 65200;   % Jun 2024
  1642,1672, 74562;   % Jul 2024
  1673,1703, 69932;   % Aug 2024
];

% Load your MONTHLY pollution data
pollution_data = readtable('cuaca_bulanan.csv'); % Replace with your actual filename

% Define which pollution variables to use (match exact column names)
pollution_vars = {'pm2p5', 'pm10', 'tcco', 'tcno2', 'gtco3', 'tcso2'}; % Adjust to match your CSV columns
num_pollution_vars = length(pollution_vars);

% Check available columns
fprintf('Available columns in pollution data:\n');
disp(pollution_data.Properties.VariableNames');

% Ensure columns exist
for i = 1:num_pollution_vars
    if ~ismember(pollution_vars{i}, pollution_data.Properties.VariableNames)
        warning('Column %s not found in pollution data!', pollution_vars{i});
    end
end

% Create a monthly pollution matrix
num_months = size(reports, 1);
monthly_pollution = zeros(num_months, num_pollution_vars);

for i = 1:num_pollution_vars
    var_name = pollution_vars{i};
    if ismember(var_name, pollution_data.Properties.VariableNames)
        % Check if we have enough data
        max_months = min(num_months, height(pollution_data));
        
        % Copy data to our monthly_pollution matrix
        monthly_pollution(1:max_months, i) = pollution_data.(var_name)(1:max_months);
    else
        warning('Variable %s not found in pollution data. Setting to NaN.', var_name);
        monthly_pollution(:, i) = NaN;
    end
end

%% 2. Create constraint matrix from reports
num_days = reports(end, 2); % Get total number of days
[A_daily, y] = rep_constraint_equations_full(reports, num_days);
placeholder_events = zeros(num_days, 1);

%% 3. Check correlations between monthly pollution and reports
monthly_correlations = zeros(num_pollution_vars, 1);
for v = 1:num_pollution_vars
    monthly_correlations(v) = corr(monthly_pollution(:, v), reports(:,3), 'rows', 'complete');
    fprintf('Correlation between monthly %s and target: %.4f\n', pollution_vars{v}, monthly_correlations(v));
end

% Scatter plots to visualize relationships
figure;
sgtitle('Monthly Pollution vs Reports');
for v = 1:num_pollution_vars
    subplot(ceil(num_pollution_vars/3), 3, v);
    scatter(monthly_pollution(:, v), reports(:,3), 'filled');
    lsline; % Add regression line
    title(sprintf('%s (r=%.2f)', pollution_vars{v}, monthly_correlations(v)));
    xlabel(pollution_vars{v});
    ylabel('Report Value');
    grid on;
end

% Find pollution variables with significant correlations
sig_threshold = 0.3; % Adjust based on your data
significant_vars = abs(monthly_correlations) >= sig_threshold;
weights = abs(monthly_correlations);
weights = weights / sum(weights(significant_vars));

%% 4. Parameter grid setup
lambdas = [0.01, 0.1, 0.5, 1, 5, 10, 50];
alphas = [0.1, 0.3, 0.5, 0.7, 0.9];
gamas = [0.3, 0.5, 0.7];
ratios = [3, 5, 7, 10];

% Create grid of parameter combinations
% We'll use a subset of combinations to keep computation manageable
selected_lambdas = [0.1, 1, 10];
selected_alphas = [0.3, 0.5, 0.7];
selected_gamas = [0.5];
selected_ratios = [3, 5, 7];

% Initialize results storage
num_combinations = length(selected_lambdas) * length(selected_alphas) * length(selected_gamas) * length(selected_ratios);
results = struct('lambda', {}, 'alpha', {}, 'gama', {}, 'ratio', {}, ...
                 'consistency_error', {}, 'monthly_correlations', {}, 'total_score', {});
results(num_combinations).lambda = 0; % Pre-allocate

%% 5. Run parameter grid search
combination_idx = 1;
total_combinations = length(selected_lambdas) * length(selected_alphas) * length(selected_gamas) * length(selected_ratios);

fprintf('Starting parameter tuning with %d combinations...\n', total_combinations);

for l_idx = 1:length(selected_lambdas)
    lambda = selected_lambdas(l_idx);
    
    for a_idx = 1:length(selected_alphas)
        alpha = selected_alphas(a_idx);
        
        for g_idx = 1:length(selected_gamas)
            gama = selected_gamas(g_idx);
            
            for r_idx = 1:length(selected_ratios)
                ratio = selected_ratios(r_idx);
                
                fprintf('Testing combination %d/%d: lambda=%.2f, alpha=%.2f, gama=%.2f, ratio=%d\n', ...
                    combination_idx, total_combinations, lambda, alpha, gama, ratio);
                
                % Create Out structure for this parameter combination
                Out = struct();
                Out(1).muvars = [30, 30]; % [Report duration, Report overlap]
                Out(1).A = A_daily;
                Out(1).y = y;
                
                % 5.1 Initial H-FUSION reconstruction
                [recon_events_daily, ~, reconstruction_error, M] = sp_reconstruct(A_daily, y, lambda, placeholder_events, [alpha]);
                Out(1).x_reconstr = recon_events_daily(:, 1);
                Out(1).Matrix = M;
                Out(1).reconst_error = reconstruction_error; % Store the error instead of expecting 'error' field
                
                % 5.2 Apply ARES filter
                [Out_A] = annihilating(Out, placeholder_events, ratio);
                
                % 5.3 Use our own iteration implementation to avoid the error
                % Instead of calling the problematic iteration function:
                % [~, ~, Out_AL] = iteration(Out, placeholder_events, 3, gama, ratio);
                
                % Custom implementation of iteration function:
                max_iter = 3; % Using 3 iterations for speed
                Inc_A = struct();
                Out_A1 = struct();
                Out_AL = struct();
                
                % First iteration
                [Inc_A(1).x_recover, ~] = reconstructas(Out(1).x_reconstr, ratio);
                Out_A1(1).x_reconstr = Inc_A(1).x_recover;
                Out_A1(1).Matrix = Out(1).Matrix;
                Out_A1(1).muvars = Out(1).muvars;
                Out_A1(1).A = Out(1).A;
                Out_A1(1).y = Out(1).y;
                Out_A1(1).reconst_error = reconstruction_error; % Store error
                
                % Store first iteration result
                Out_AL = Out_A1;
                
                % Subsequent iterations
                for iter = 2:max_iter
                    % Apply H-FUSION with updated parameters
                    [recon_events_iter, ~, iter_error, M_iter] = sp_reconstruct(A_daily, y, lambda * gama, Out_AL(1).x_reconstr, [alpha]);
                    
                    % Update with new reconstruction
                    Out_AL(1).x_reconstr = recon_events_iter(:, 1);
                    Out_AL(1).Matrix = M_iter;
                    Out_AL(1).reconst_error = iter_error;
                    
                    % Apply ARES filter again
                    [Inc_A(iter).x_recover, ~] = reconstructas(Out_AL(1).x_reconstr, ratio);
                    Out_AL(1).x_reconstr = Inc_A(iter).x_recover;
                end
                
                % 5.4 Evaluate this combination
                % a) Check report consistency (reconstructed values should sum to reported totals)
                consistency_errors = zeros(size(reports, 1), 1);
                for i = 1:size(reports, 1)
                    from = reports(i,1);
                    to = reports(i,2);
                    reported = reports(i,3);
                    reconstructed = sum(Out_AL(1).x_reconstr(from:to));
                    consistency_errors(i) = abs(reported - reconstructed) / reported; % Relative error
                end
                mean_consistency_error = mean(consistency_errors);
                
                % b) Aggregate reconstructed daily data to monthly for comparison with pollution
                recon_monthly = zeros(size(reports, 1), 1);
                for i = 1:size(reports, 1)
                    from = reports(i,1);
                    to = reports(i,2);
                    recon_monthly(i) = mean(Out_AL(1).x_reconstr(from:to));
                end
                
                % c) Check correlation with all pollution variables at the monthly level
                recon_correlations = zeros(num_pollution_vars, 1);
                corr_differences = zeros(num_pollution_vars, 1);
                
                for v = 1:num_pollution_vars
                    % Calculate correlation between reconstructed monthly data and pollution
                    recon_correlations(v) = corr(recon_monthly, monthly_pollution(:, v), 'rows', 'complete');
                    
                    % Compare with original correlations
                    corr_differences(v) = abs(recon_correlations(v) - monthly_correlations(v));
                end
                
                % Calculate weighted correlation score (only using significant variables)
                weighted_corr_score = 0;
                if sum(significant_vars) > 0
                    for v = 1:num_pollution_vars
                        if significant_vars(v)
                            weighted_corr_score = weighted_corr_score + weights(v) * corr_differences(v);
                        end
                    end
                else
                    % If no significant correlations, use mean of all differences
                    weighted_corr_score = mean(corr_differences);
                end
                
                % d) Calculate combined score (lower is better)
                combined_score = mean_consistency_error + weighted_corr_score;
                
                % 5.5 Store results
                results(combination_idx).lambda = lambda;
                results(combination_idx).alpha = alpha;
                results(combination_idx).gama = gama;
                results(combination_idx).ratio = ratio;
                results(combination_idx).consistency_error = mean_consistency_error;
                results(combination_idx).correlation_score = weighted_corr_score;
                results(combination_idx).total_score = combined_score;
                
                % Store individual pollution variable correlations
                results(combination_idx).recon_correlations = recon_correlations;
                
                combination_idx = combination_idx + 1;
            end
        end
    end
end

%% 6. Analyze results and find optimal parameters
% Convert to table for easier analysis
results_table = struct2table(results);

% Sort by combined score (lower is better)
sorted_results = sortrows(results_table, 'total_score');

% Display top 5 parameter combinations
disp('Top 5 parameter combinations:');
disp(sorted_results(1:min(5,height(sorted_results)), {'lambda', 'alpha', 'gama', 'ratio', 'consistency_error', 'correlation_score', 'total_score'}));

% Extract optimal parameters
best_lambda = sorted_results.lambda(1);
best_alpha = sorted_results.alpha(1);
best_gama = sorted_results.gama(1);
best_ratio = sorted_results.ratio(1);

fprintf('\nOptimal parameters found:\n');
fprintf('Lambda: %.2f\n', best_lambda);
fprintf('Alpha: %.2f\n', best_alpha);
fprintf('Gama: %.2f\n', best_gama);
fprintf('Ratio: %d\n', best_ratio);

%% 7. Run final reconstruction with optimal parameters
fprintf('\nRunning final reconstruction with optimal parameters...\n');

% Create Out structure with optimal parameters
Out = struct();
Out(1).muvars = [30, 30];
Out(1).A = A_daily;
Out(1).y = y;

% Initial H-FUSION reconstruction
[recon_events_daily, ~, reconstruction_error, M] = sp_reconstruct(A_daily, y, best_lambda, placeholder_events, [best_alpha]);
Out(1).x_reconstr = recon_events_daily(:, 1);
Out(1).Matrix = M;
Out(1).reconst_error = reconstruction_error;

% Apply ARES filter
[Out_A] = annihilating(Out, placeholder_events, best_ratio);

% Use our own implementation of iteration for the final run
max_iter = 5; % More iterations for final run
Inc_A = struct();
Out_A1 = struct();
Out_AL = struct();

% First iteration
[Inc_A(1).x_recover, ~] = reconstructas(Out(1).x_reconstr, best_ratio);
Out_A1(1).x_reconstr = Inc_A(1).x_recover;
Out_A1(1).Matrix = Out(1).Matrix;
Out_A1(1).muvars = Out(1).muvars;
Out_A1(1).A = Out(1).A;
Out_A1(1).y = Out(1).y;
Out_A1(1).reconst_error = reconstruction_error;

% Store first iteration result
Out_AL = Out_A1;

% Subsequent iterations
for iter = 2:max_iter
    % Apply H-FUSION with updated parameters
    [recon_events_iter, ~, iter_error, M_iter] = sp_reconstruct(A_daily, y, best_lambda * best_gama, Out_AL(1).x_reconstr, [best_alpha]);
    
    % Update with new reconstruction
    Out_AL(1).x_reconstr = recon_events_iter(:, 1);
    Out_AL(1).Matrix = M_iter;
    Out_AL(1).reconst_error = iter_error;
    
    % Apply ARES filter again
    [Inc_A(iter).x_recover, ~] = reconstructas(Out_AL(1).x_reconstr, best_ratio);
    Out_AL(1).x_reconstr = Inc_A(iter).x_recover;
end

%% 8. Visualize and save final results
figure;

% Plot H-FUSION reconstruction
subplot(3,1,1);
plot(Out(1).x_reconstr);
title('H-FUSION Reconstruction');
ylabel('Daily Counts');
grid on;

% Plot first ARES iteration
subplot(3,1,2);
plot(Out_A1(1).x_reconstr);
title('ARES - First Iteration');
ylabel('Daily Counts');
grid on;

% Plot final ARES iteration
subplot(3,1,3);
plot(Out_AL(1).x_reconstr);
title('ARES - Final Iteration (Optimal Parameters)');
xlabel('Day');
ylabel('Daily Counts');
grid on;

% Regenerate monthly aggregation from final reconstruction
recon_monthly = zeros(size(reports, 1), 1);
for i = 1:size(reports, 1)
    from = reports(i,1);
    to = reports(i,2);
    recon_monthly(i) = mean(Out_AL(1).x_reconstr(from:to));
end

% Plot comparison of monthly data
figure;
subplot(2,1,1);
bar([reports(:,3), recon_monthly]);
title('Original vs Reconstructed Monthly Data');
legend('Original Reports', 'Reconstructed');
ylabel('Monthly Value');
grid on;

subplot(2,1,2);
plot(reports(:,3) - recon_monthly);
title('Reconstruction Error (Original - Reconstructed)');
xlabel('Month');
ylabel('Error');
grid on;

% Plot correlation with significant pollution variables
% First, recalculate final correlations
final_recon_correlations = zeros(num_pollution_vars, 1);
for v = 1:num_pollution_vars
    final_recon_correlations(v) = corr(recon_monthly, monthly_pollution(:, v), 'rows', 'complete');
end

figure;
bar([monthly_correlations, final_recon_correlations]);
title('Correlation Comparison: Original vs Reconstructed');
set(gca, 'XTick', 1:num_pollution_vars, 'XTickLabel', pollution_vars);
xtickangle(45);
ylabel('Correlation');
legend('Original', 'Reconstructed');
grid on;

% Plot scatter for significant variables
figure;
significant_count = sum(significant_vars);
if significant_count > 0
    for v = 1:num_pollution_vars
        if significant_vars(v)
            subplot(ceil(significant_count/2), 2, find(significant_vars == 1, v));
            scatter(monthly_pollution(:, v), recon_monthly, 'filled');
            lsline;
            title(sprintf('%s vs Recon (r=%.2f)', pollution_vars{v}, final_recon_correlations(v)));
            xlabel(pollution_vars{v});
            ylabel('Reconstructed Monthly');
            grid on;
        end
    end
end

% Save results
optimal_parameters = struct('lambda', best_lambda, 'alpha', best_alpha, ...
                           'gama', best_gama, 'ratio', best_ratio);
save('optimal_reconstruction_results.mat', 'Out', 'Out_A1', 'Out_AL', 'Inc_A', ...
     'reports', 'optimal_parameters', 'results_table', 'monthly_correlations', 'final_recon_correlations');
fprintf('\nResults saved to optimal_reconstruction_results.mat\n');

%% 9. Evaluate final reconstruction quality
fprintf('\nFinal reconstruction quality evaluation:\n');
fprintf('Consistency error: %.4f\n', sorted_results.consistency_error(1));
fprintf('Correlation quality score: %.4f\n', sorted_results.correlation_score(1));

% Print correlation comparisons for each pollution variable
fprintf('\nCorrelations between monthly data and pollution variables:\n');
fprintf('%-10s | %-15s | %-15s | %-10s\n', 'Variable', 'Original', 'Reconstructed', 'Difference');
fprintf('%-10s | %-15s | %-15s | %-10s\n', '--------', '---------------', '---------------', '----------');
for v = 1:num_pollution_vars
    fprintf('%-10s | %-15.4f | %-15.4f | %-10.4f\n', ...
        pollution_vars{v}, monthly_correlations(v), final_recon_correlations(v), ...
        abs(monthly_correlations(v) - final_recon_correlations(v)));
end