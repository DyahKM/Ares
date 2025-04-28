% Modified main.m for Monthly to Daily Disaggregation
% Purpose: Convert monthly reports to daily estimates while maintaining the experimental structure
% Period covered: January 2020 - November 2024

addpath('HFusion')
addpath('AresFilter')
addpath('data')
clc; clear;

%% 1. Define monthly reports (Jan 2020 - Nov 2024)
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
  1461,1491, 74898;   % Jan 2024
  1492,1519, 65049;   % Feb 2024 (29 days, leap year)
  1520,1550, 67988;   % Mar 2024
  1551,1580, 66322;   % Apr 2024
  1581,1611, 75951;   % May 2024
  1612,1641, 65180;   % Jun 2024
  1642,1672, 74885;   % Jul 2024
  1673,1703, 71239;   % Aug 2024
  1704,1733, 70057;   % Sep 2024 (Fixed index from 1706 to 1704)
  1734,1764, 77470;   % Oct 2024 (Fixed index from 1736 to 1734)
  1765,1794, 70135;   % Nov 2024 (Fixed index from 1767 to 1765)
];

% Create placeholder events vector for the full time period
num_days = reports(end, 2);
events = zeros(num_days, 1);

%% 2. Setup parameters
% H-FUSION Parameters
lambdas = [0.1, 1, 5, 10, 20];  % Multiple lambda values to test
alpha = 0.5;                    % Balance between smoothness and periodicity
gama = 0.5;                     % Parameter for cost function
iterationTime = 4;              % Number of iterations

% Configure different experimental setups for report aggregation tests
% We'll use sliding windows of reports with different durations and overlaps
% For your monthly data use case, this is primarily for testing the robustness of the method
config_rep_dur = [15, 30, 60];  % Report durations: half-month, month, 2-months
config_rep_over = [1, 7, 15];   % Report overlaps: daily, weekly, bi-weekly
xdim = length(config_rep_dur);
ydim = length(config_rep_over);

% Save initial setup variables
save('TB_disaggregation_setup.mat', 'reports', 'events', 'lambdas', 'alpha', 'gama', 'iterationTime', ...
    'config_rep_dur', 'config_rep_over', 'xdim', 'ydim');

%% 3. First Phase: Initial reconstruction using H-Fusion
fprintf('Starting H-Fusion reconstruction...\n');

% Get constraint matrix from the reports
[A, y] = rep_constraint_equations_full(reports, events);

% Create initial reconstruction using original reports
Out_original = struct();
Out_original(1).muvars = [0, 0];  % Special indicator for original reports
Out_original(1).A = A;
Out_original(1).y = y;

[recon_events, recon_error, reconstruction_param, M] = sp_reconstruct(A, y, lambdas, events, alpha);
Out_original(1).x_reconstr = recon_events(:, 1, 1);  % Use first lambda and alpha values
Out_original(1).x_error = recon_error;
Out_original(1).Matrix = M;
Out_original(1).sp_params = reconstruction_param;
[Out_original(1).error, Out_original(1).minIdx] = min(recon_error(:));

% Save the first phase results
save('TB_phase1_original.mat', 'A', 'y', 'Out_original', 'recon_events', 'recon_error', 'reconstruction_param', 'M');

% Also create experimental synthetic reports of different durations to test robustness
fprintf('Running H-Fusion with different report configurations...\n');
[Out, Out_LSQ] = hfusion(events, lambdas, alpha, config_rep_dur, config_rep_over);

% Combine original reports results with synthetic test results
Out = [Out_original, Out];

% Save the combined results
save('TB_phase1_combined.mat', 'Out', 'Out_LSQ');

%% 4. Second Phase: Enhance reconstruction using Annihilating Filter
fprintf('Starting ARES filter processing...\n');

% Try different annihilating filter ratios
stop_ratios = [2, 5, 8];
RMSE = zeros(length(stop_ratios), iterationTime+1);

% Container for results from each ratio
All_Inc_A = cell(length(stop_ratios), 1);
All_Out_A1 = cell(length(stop_ratios), 1);
All_Out_AL = cell(length(stop_ratios), 1);

% For each ratio, run the iteration process
for i = 1:length(stop_ratios)
    fprintf('Running ARES with stop ratio %.2f\n', stop_ratios(i));
    
    % Run iterative refinement with annihilating filter
    [Inc_A, Out_A1, Out_AL] = iteration(Out, events, iterationTime, gama, stop_ratios(i));
    
    % Store in the containers
    All_Inc_A{i} = Inc_A;
    All_Out_A1{i} = Out_A1;
    All_Out_AL{i} = Out_AL;
    
    % Save results for this ratio
    file_name = sprintf('TB_ratio_%.2f.mat', stop_ratios(i));
    save(file_name, 'Inc_A', 'Out_A1', 'Out_AL', 'stop_ratios', 'i');
    
    % Record RMSE progression
    RMSE(i,:) = mean(Inc_A.ActError);
    
    % For the last ratio, extract and save the daily estimates
    if i == length(stop_ratios)
        daily_estimates = Out_AL(1).x_reconstr;
        
        % Enforce the monthly constraints with integer values
        daily_estimates_constrained = enforce_monthly_integer_constraints(daily_estimates, reports);
        
        % Save constrained and unconstrained estimates
        save('TB_daily_estimates.mat', 'daily_estimates', 'daily_estimates_constrained', 'reports');
        
        % Save daily estimates to CSV for easier access
        dates = (1:length(daily_estimates_constrained))';
        daily_output = [dates, daily_estimates_constrained];
        csvwrite('TB_daily_estimates.csv', daily_output);
        
        % Plot results for constrained estimates
        figure;
        plot(daily_estimates_constrained, 'LineWidth', 1.5);
        title('Estimated Daily TB Cases (Integer Constraint Enforced)');
        xlabel('Day (1 = Jan 1, 2020)');
        ylabel('Number of TB Cases');
        grid on;
        saveas(gcf, 'TB_daily_estimates_integer.png');
        
        % Also plot monthly sums to verify constraints are preserved
        monthly_sums = zeros(size(reports,1), 1);
        for j = 1:size(reports,1)
            monthly_sums(j) = sum(daily_estimates_constrained(reports(j,1):reports(j,2)));
        end
        
        % Calculate and display the constraint errors
        original_sums = zeros(size(reports,1), 1);
        for j = 1:size(reports,1)
            original_sums(j) = sum(daily_estimates(reports(j,1):reports(j,2)));
        end
        
        max_error_before = max(abs(original_sums - reports(:,3))) / max(reports(:,3)) * 100;
        max_error_after = max(abs(monthly_sums - reports(:,3))) / max(reports(:,3)) * 100;
        
        fprintf('Maximum constraint error before adjustment: %.6f%%\n', max_error_before);
        fprintf('Maximum constraint error after adjustment: %.6f%%\n', max_error_after);
        
        % Check if all daily estimates are integers
        if all(daily_estimates_constrained == round(daily_estimates_constrained))
            fprintf('All daily estimates are integers as required for TB case counts.\n');
        else
            fprintf('WARNING: Not all daily estimates are integers!\n');
        end
        
        % Save verification data
        verification_data = struct();
        verification_data.monthly_sums = monthly_sums;
        verification_data.original_sums = original_sums;
        verification_data.max_error_before = max_error_before;
        verification_data.max_error_after = max_error_after;
        verification_data.all_integers = all(daily_estimates_constrained == round(daily_estimates_constrained));
        save('TB_verification.mat', 'verification_data');
        
        figure;
        bar([reports(:,3), monthly_sums, original_sums]);
        title('Monthly Reports vs. Sum of Daily Estimates');
        xlabel('Month Index');
        ylabel('Number of TB Cases');
        legend('Original Monthly Reports', 'Constrained Daily Sums', 'Unconstrained Daily Sums');
        grid on;
        saveas(gcf, 'TB_monthly_verification.png');
        
        % Also save comparison to CSV
        monthly_comparison = [reports(:,3), monthly_sums, original_sums, reports(:,3) - monthly_sums, reports(:,3) - original_sums];
        csvwrite('TB_monthly_comparison.csv', monthly_comparison);
        
        % Create detailed daily report for first few months as example
        figure;
        months_to_show = min(6, size(reports,1));
        end_day = reports(months_to_show, 2);
        plot(1:end_day, daily_estimates_constrained(1:end_day), 'LineWidth', 1.5);
        title('Daily TB Cases - First 6 Months');
        xlabel('Day');
        ylabel('Number of TB Cases');
        grid on;
        
        % Add vertical lines and labels to separate months
        hold on;
        for j = 1:months_to_show
            if j > 1
                xline(reports(j,1)-0.5, '--', 'LineWidth', 1);
            end
            month_midpoint = (reports(j,1) + reports(j,2))/2;
            text(month_midpoint, max(daily_estimates_constrained(1:end_day))*0.9, ...
                sprintf('Month %d', j), 'HorizontalAlignment', 'center');
        end
        hold off;
        saveas(gcf, 'TB_daily_detail_first_months.png');
        
        % Save the detailed month visualization data
        detail_plot_data = struct();
        detail_plot_data.months_to_show = months_to_show;
        detail_plot_data.end_day = end_day;
        detail_plot_data.daily_data = daily_estimates_constrained(1:end_day);
        save('TB_detail_plot_data.mat', 'detail_plot_data');
    end
end

% Combine all results
save('TB_all_results.mat', 'All_Inc_A', 'All_Out_A1', 'All_Out_AL', 'RMSE', 'stop_ratios');

% Plot RMSE comparison for different stop ratios
figure;
plot(0:iterationTime, RMSE', 'LineWidth', 1.5);
title('RMSE Progression with Different Stop Ratios');
xlabel('Iteration');
ylabel('RMSE');
legend(arrayfun(@(x) sprintf('Ratio %.2f', x), stop_ratios, 'UniformOutput', false));
grid on;
saveas(gcf, 'TB_rmse_comparison.png');

% Save everything in the workspace for future use
save('TB_complete_workspace.mat');

fprintf('Process complete. All variables saved to "TB_complete_workspace.mat"\n');
fprintf('Integer daily estimates saved to "TB_daily_estimates.csv"\n');
fprintf('Monthly comparison saved to "TB_monthly_comparison.csv"\n');

%% 5. Function to enforce monthly constraints with integer values
function constrained_estimates = enforce_monthly_integer_constraints(daily_estimates, reports)
    % Initialize constrained estimates with the original estimates
    constrained_estimates = daily_estimates;
    
    % For each month, adjust the daily values to match the monthly total using integers
    for i = 1:size(reports, 1)
        start_day = reports(i, 1);
        end_day = reports(i, 2);
        target_sum = reports(i, 3);
        
        % Get the current estimates for this month
        month_estimates = daily_estimates(start_day:end_day);
        
        % First approach: Use proportional rounding to integers while preserving sum
        adjusted_values = dither_to_integers(month_estimates, target_sum);
        
        % Assign the integer values back to the constrained estimates
        constrained_estimates(start_day:end_day) = adjusted_values;
    end
end

%% 6. Helper function for integer dithering while preserving sum
function int_values = dither_to_integers(values, target_sum)
    % Scale the values to match the target sum
    scaling_factor = target_sum / sum(values);
    scaled_values = values * scaling_factor;
    
    % Initial rounding (will likely not preserve the sum exactly)
    int_values = round(scaled_values);
    current_sum = sum(int_values);
    
    % Calculate the difference we need to distribute
    diff = target_sum - current_sum;
    
    if diff == 0
        % Sum is already correct, no adjustment needed
        return;
    end
    
    % Find fractional parts to determine where to adjust
    fractional_parts = scaled_values - floor(scaled_values);
    
    if diff > 0
        % Need to add 'diff' more counts
        % Sort by the fractional part in descending order
        % (values closer to being rounded up are adjusted first)
        [~, idx] = sort(fractional_parts, 'descend');
        for i = 1:diff
            int_values(idx(i)) = int_values(idx(i)) + 1;
        end
    else
        % Need to subtract 'diff' counts
        % Sort by the fractional part in ascending order
        % (values closer to being rounded down are adjusted first)
        [~, idx] = sort(fractional_parts, 'ascend');
        for i = 1:abs(diff)
            int_values(idx(i)) = int_values(idx(i)) - 1;
        end
    end
    
    % Ensure no negative values
    negative_indices = find(int_values < 0);
    if ~isempty(negative_indices)
        % Calculate how many negative values we have
        neg_sum = sum(int_values(negative_indices));
        % Set them to zero
        int_values(negative_indices) = 0;
        % We need to compensate by decreasing other values
        positive_indices = find(int_values > 0);
        
        % Sort by value in descending order to take from the largest values first
        [sorted_vals, idx] = sort(int_values(positive_indices), 'descend');
        pos_idx = positive_indices(idx);
        
        % Distribute the negative sum among positive values
        adjust_idx = 1;
        remaining = abs(neg_sum);
        
        while remaining > 0 && adjust_idx <= length(pos_idx)
            if int_values(pos_idx(adjust_idx)) > 1  % Keep at least 1
                int_values(pos_idx(adjust_idx)) = int_values(pos_idx(adjust_idx)) - 1;
                remaining = remaining - 1;
            end
            
            adjust_idx = adjust_idx + 1;
            if adjust_idx > length(pos_idx) && remaining > 0
                % If we need to continue reducing, start again from largest values
                adjust_idx = 1;
            end
        end
    end
end
