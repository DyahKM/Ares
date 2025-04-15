% Main script for disaggregating monthly reports into daily values
% with exact monthly sum preservation
% Period: January 2020 - November 2024

addpath('HFusion')
addpath('AresFilter')
clc; clear;

%% 1. Define the monthly reports (from Jan 2020 to Nov 2024)
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
  1706,1735, 70057;   % Sep 2024
  1736,1766, 77470;   % Oct 2024
  1767,1796, 70135;   % Nov 2024
];

%% 2. Setup parameters for the reconstruction
% H-FUSION Parameters
lambda = 5;        % Regularization parameter
alpha = 0.5;       % Balance between smoothness and periodicity

% ARES Parameters
gama = 0.5;        % Weight for cost function
ratio = 5;         % Annihilating filter length ratio
iterationTime = 4; % Number of iterations

% Create placeholder events (will be updated with our reconstruction)
num_days = reports(end, 2);
placeholder_events = zeros(num_days, 1);

%% 3. Create constraint matrix from monthly reports
fprintf('Creating constraint matrix from %d monthly reports...\n', size(reports, 1));
[A, y] = rep_constraint_equations_full(reports, placeholder_events);

%% 4. Initial reconstruction with H-FUSION
fprintf('Performing initial reconstruction with H-FUSION...\n');
Out = struct();
Out(1).muvars = [30, 30]; % [Report duration, Report overlap]
Out(1).A = A;
Out(1).y = y;

[recon_events, ~, reconstruction_param, M] = sp_reconstruct(A, y, lambda, placeholder_events, alpha);
Out(1).x_reconstr = recon_events(:, 1, 1);
Out(1).Matrix = M;
Out(1).error = sqrt(mean((recon_events(:, 1, 1) - placeholder_events).^2));

%% 5. Apply ARES filter
fprintf('Applying ARES filter...\n');
[Out_A] = annihilating(Out, placeholder_events, ratio);

%% 6. Iterate between H-FUSION and ARES
fprintf('Performing %d iterations of H-FUSION and ARES...\n', iterationTime);
[Inc_A, Out_A1, Out_AL] = iteration(Out, placeholder_events, iterationTime, gama, ratio);

%% 7. Round the daily values while preserving monthly sums with minimal pattern distortion
fprintf('Rounding daily values while preserving monthly sums and original patterns...\n');
daily_values = Out_AL(1).x_reconstr;

% First, ensure no negative values
daily_values = max(daily_values, 0);

% Initialize rounded values
rounded_values = zeros(size(daily_values));

% Set maximum allowed deviation as a percentage of the original value
max_allowed_deviation_pct = 25; % Maximum percentage change from original

% Process each month separately
for i = 1:size(reports, 1)
    from = reports(i, 1);
    to = reports(i, 2);
    target_sum = reports(i, 3);
    
    % Extract values for this month
    month_values = daily_values(from:to);
    days_in_month = length(month_values);
    
    % Step 1: Handle very small values but minimize distortion
    % Calculate mean and set minimum threshold based on it
    month_mean = mean(month_values);
    small_threshold = 0.5; % Values below this will be adjusted
    
    % Find very small values
    very_small_indices = find(month_values < small_threshold);
    
    if ~isempty(very_small_indices)
        % Calculate a scaling factor to lift small values while preserving pattern
        min_val = min(month_values);
        if min_val < 0.01 % Avoid division by zero or very small numbers
            min_val = 0.01;
        end
        
        % Scale only the very small values
        boost_factor = small_threshold / min_val;
        
        % Apply a gradual boost to small values
        for j = 1:length(very_small_indices)
            idx = very_small_indices(j);
            original = month_values(idx);
            
            % Boost small values proportionally, more boost for smaller values
            if original < 0.1
                month_values(idx) = small_threshold;
            elseif original < small_threshold
                % Gradually boost based on how small the value is
                boost_amount = (small_threshold - original) * (original / small_threshold);
                month_values(idx) = original + boost_amount;
            end
        end
        
        % Re-scale to maintain total sum
        scale_factor = target_sum / sum(month_values);
        month_values = month_values * scale_factor;
    end
    
    % Step 2: Initial rounding
    rounded_month = round(month_values);
    initial_rounded = rounded_month; % Save for comparison
    
    % Step 3: Adjust to match target sum
    current_sum = sum(rounded_month);
    diff = target_sum - current_sum;
    
    if diff ~= 0
        % Calculate fractional parts and distance from rounding boundary
        if diff > 0
            % Need to round up more values - look at values rounded down
            candidates = find(rounded_month < month_values);
            distances = month_values(candidates) - rounded_month(candidates);
        else
            % Need to round down more values - look at values rounded up
            candidates = find(rounded_month > month_values);
            distances = rounded_month(candidates) - month_values(candidates);
        end
        
        % If we don't have enough candidates, use all values
        if length(candidates) < abs(diff)
            if diff > 0
                % Need more round-ups: sort by fractional part (descending)
                [~, idx] = sort(month_values - floor(month_values), 'descend');
            else
                % Need more round-downs: sort by fractional part (ascending)
                [~, idx] = sort(month_values - floor(month_values), 'ascend');
            end
        else
            % Sort candidates by distance (ascending)
            [~, sorted_idx] = sort(distances);
            idx = candidates(sorted_idx);
        end
        
        % Apply adjustments
        adjustments_made = 0;
        j = 1;
        
        while adjustments_made < abs(diff) && j <= length(idx)
            adjust_idx = idx(j);
            
            % Calculate what the value would be after adjustment
            new_value = rounded_month(adjust_idx) + sign(diff);
            
            % Check if adjustment would create too large a deviation
            original = month_values(adjust_idx);
            
            if original > 0
                deviation_pct = 100 * abs(new_value - original) / original;
                
                % Only adjust if deviation is acceptable
                if deviation_pct <= max_allowed_deviation_pct
                    rounded_month(adjust_idx) = new_value;
                    adjustments_made = adjustments_made + 1;
                end
            else
                % For zero or near-zero values, just set to 1 if needed
                if diff > 0 && rounded_month(adjust_idx) == 0
                    rounded_month(adjust_idx) = 1;
                    adjustments_made = adjustments_made + 1;
                end
            end
            
            j = j + 1;
            
            % If we've gone through all candidates but still need adjustments,
            % relax the deviation constraint
            if j > length(idx) && adjustments_made < abs(diff)
                max_allowed_deviation_pct = max_allowed_deviation_pct * 1.5;
                j = 1; % Start over with relaxed constraint
                
                % Safety check to prevent infinite loop
                if max_allowed_deviation_pct > 200
                    fprintf('Warning: Had to allow large deviations in month %d\n', i);
                    
                    % Force the remaining adjustments on largest/smallest values
                    remaining = abs(diff) - adjustments_made;
                    
                    if diff > 0
                        % Need to increase more values
                        [~, idx_vals] = sort(rounded_month);
                        for k = 1:remaining
                            if k <= length(idx_vals)
                                rounded_month(idx_vals(k)) = rounded_month(idx_vals(k)) + 1;
                            end
                        end
                    else
                        % Need to decrease more values
                        [~, idx_vals] = sort(rounded_month, 'descend');
                        for k = 1:remaining
                            if k <= length(idx_vals) && rounded_month(idx_vals(k)) > 1
                                rounded_month(idx_vals(k)) = rounded_month(idx_vals(k)) - 1;
                            end
                        end
                    end
                    
                    break;
                end
            end
        end
    end
    
    % Step 4: Final check for zeros
    zero_indices = find(rounded_month == 0);
    
    if ~isempty(zero_indices)
        fprintf('Month %d has %d zeros after adjustments. Fixing...\n', i, length(zero_indices));
        
        % Fix each zero while trying to minimize pattern distortion
        for j = 1:length(zero_indices)
            zero_idx = zero_indices(j);
            
            % Find the closest non-zero values (time-wise) to maintain pattern
            distances = abs((1:length(rounded_month)) - zero_idx);
            [~, idx_sorted] = sort(distances);
            
            donor_found = false;
            
            % Try to find a nearby value > 1 to take from
            for k = 1:length(idx_sorted)
                check_idx = idx_sorted(k);
                
                if rounded_month(check_idx) > 1
                    % Found a donor
                    rounded_month(check_idx) = rounded_month(check_idx) - 1;
                    rounded_month(zero_idx) = 1;
                    donor_found = true;
                    break;
                end
            end
            
            if ~donor_found
                % No suitable donor found - just set to 1 and handle sum later
                rounded_month(zero_idx) = 1;
                
                % Find largest value to decrease
                [max_val, max_idx] = max(rounded_month);
                if max_val > 1 && max_idx ~= zero_idx
                    rounded_month(max_idx) = rounded_month(max_idx) - 1;
                end
            end
        end
    end
    
    % Calculate deviation statistics for this month
    original_month = month_values;
    abs_diffs = abs(rounded_month - original_month);
    rel_diffs = zeros(size(abs_diffs));
    
    for j = 1:length(original_month)
        if original_month(j) > 0
            rel_diffs(j) = 100 * abs_diffs(j) / original_month(j);
        else
            rel_diffs(j) = 0;
        end
    end
    
    max_abs_diff = max(abs_diffs);
    max_rel_diff = max(rel_diffs);
    mean_abs_diff = mean(abs_diffs);
    mean_rel_diff = mean(rel_diffs);
    
    fprintf('Month %d stats - Max abs diff: %.2f, Max rel diff: %.2f%%%%, Mean abs diff: %.2f, Mean rel diff: %.2f%%%%\n', i, max_abs_diff, max_rel_diff, mean_abs_diff, mean_rel_diff);
    
    % Store the adjusted values
    rounded_values(from:to) = rounded_month;
    
    % Final verification
    actual_sum = sum(rounded_values(from:to));
    if actual_sum ~= target_sum
        fprintf('Warning: Month %d sum mismatch: target=%d, actual=%d, diff=%d\n', i, target_sum, actual_sum, target_sum - actual_sum);
    end
end

% Calculate overall statistics
abs_diffs = abs(rounded_values - daily_values);
rel_diffs = zeros(size(abs_diffs));

for i = 1:length(daily_values)
    if daily_values(i) > 0
        rel_diffs(i) = 100 * abs_diffs(i) / daily_values(i);
    else
        rel_diffs(i) = 0;
    end
end

% Find indices with large relative differences
large_diff_threshold = 50; % 50% difference
large_diff_indices = find(rel_diffs > large_diff_threshold);

% Find indices with large absolute differences
large_abs_threshold = 100;
large_abs_indices = find(abs_diffs > large_abs_threshold);

% Special check for indices 1704 and 1705
fprintf('\nSpecial check for indices 1704 and 1705:\n');
for idx = [1704, 1705]
    original = daily_values(idx);
    rounded = rounded_values(idx);
    abs_diff = abs(rounded - original);
    
    if original > 0
        rel_diff = 100 * abs_diff / original;
    else
        rel_diff = inf;
    end
    
    fprintf('Index %d - Original: %.4f, Rounded: %d, Abs Diff: %.4f, Rel Diff: %.2f%%\n', idx, original, rounded, abs_diff, rel_diff);
    
    % Find which month this index belongs to
    for i = 1:size(reports, 1)
        if reports(i,1) <= idx && idx <= reports(i,2)
            fprintf('  This index belongs to month %d (days %d-%d)\n', i, reports(i,1), reports(i,2));
            break;
        end
    end
end

%% Show values around problem indices
fprintf('\nValues around problem indices 1704 and 1705:\n');
fprintf('%-6s %-12s %-12s %-12s %-12s\n', 'Index', 'Original', 'Rounded', 'Abs Diff', 'Rel Diff (%)');
fprintf('%-6s %-12s %-12s %-12s %-12s\n', '------', '------------', '------------', '------------', '------------');

for idx = 1699:1710
    original = daily_values(idx);
    rounded = rounded_values(idx);
    abs_diff = abs(rounded - original);
    
    if original > 0
        rel_diff = 100 * abs_diff / original;
    else
        rel_diff = inf;
    end
    
    fprintf('%-6d %-12.4f %-12d %-12.4f %-12.2f\n', idx, original, rounded, abs_diff, rel_diff);
end

% Overall deviation statistics
fprintf('\nOverall deviation statistics:\n');
fprintf('Mean absolute difference: %.4f\n', mean(abs_diffs));
fprintf('Mean relative difference: %.2f%%\n', mean(rel_diffs));
fprintf('Maximum absolute difference: %.4f\n', max(abs_diffs));
fprintf('Maximum relative difference: %.2f%%\n', max(rel_diffs));
fprintf('Number of values with rel diff > %d%%: %d (%.2f%% of total)\n', large_diff_threshold, length(large_diff_indices), 100*length(large_diff_indices)/length(daily_values));
fprintf('Number of values with abs diff > %d: %d (%.2f%% of total)\n', large_abs_threshold, length(large_abs_indices), 100*length(large_abs_indices)/length(daily_values));

% Check for any zero values in the final result
if any(rounded_values == 0)
    fprintf('\nWarning: The final result still contains %d zero values\n', sum(rounded_values == 0));
    zero_indices = find(rounded_values == 0);
    fprintf('Zero values are found at indices: ');
    fprintf('%d ', zero_indices(1:min(10, length(zero_indices))));
    if length(zero_indices) > 10
        fprintf('... (and %d more)', length(zero_indices) - 10);
    end
    fprintf('\n');
else
    fprintf('\nSuccess: No zero values in the final result\n');
end

%% Verify all monthly sums
fprintf('\nFinal verification of monthly sums:\n');
all_match = true;
mismatch_months = [];

for i = 1:size(reports, 1)
    from = reports(i, 1);
    to = reports(i, 2);
    target_sum = reports(i, 3);
    actual_sum = sum(rounded_values(from:to));
    
    if actual_sum ~= target_sum
        all_match = false;
        mismatch_months(end+1) = i;
        fprintf('Month %d: Target=%d, Actual=%d, Diff=%d\n',i, target_sum, actual_sum, target_sum - actual_sum);
    end
end

if all_match
    fprintf('Success: All monthly sums exactly match the targets.\n');
else
    fprintf('Warning: %d months have sum mismatches\n', length(mismatch_months));
end
%% % After processing all months, check specifically for indices 1704 and 1705
if rounded_values(1704) == 0
    fprintf('Fixing anomalous zero at index 1704\n');
    rounded_values(1704) = round(daily_values(1704)); % Just use the rounded value
    
    % Find which month this index belongs to
    for i = 1:size(reports, 1)
        if reports(i,1) <= 1704 && 1704 <= reports(i,2)
            % Adjust another value in the same month to maintain sum
            month_idx = i;
            from = reports(i, 1);
            to = reports(i, 2);
            
            % Find a large value to reduce by 1
            [~, max_idx] = max(rounded_values(from:to));
            max_idx = max_idx + from - 1; % Convert to global index
            
            if max_idx ~= 1704 % Make sure we're not reducing the one we just fixed
                rounded_values(max_idx) = rounded_values(max_idx) - rounded_values(1704);
                fprintf('Balanced by adjusting index %d\n', max_idx);
            else
                % Find second largest
                temp_values = rounded_values(from:to);
                temp_values(max_idx-from+1) = 0; % Zero out the max
                [~, second_max_idx] = max(temp_values);
                second_max_idx = second_max_idx + from - 1; % Convert to global index
                rounded_values(second_max_idx) = rounded_values(second_max_idx) - rounded_values(1704);
                fprintf('Balanced by adjusting index %d\n', second_max_idx);
            end
            break;
        end
    end
end

if rounded_values(1705) == 0
    fprintf('Fixing anomalous zero at index 1705\n');
    rounded_values(1705) = round(daily_values(1705)); % Just use the rounded value
    
    % Find which month this index belongs to
    for i = 1:size(reports, 1)
        if reports(i,1) <= 1705 && 1705 <= reports(i,2)
            % Adjust another value in the same month to maintain sum
            month_idx = i;
            from = reports(i, 1);
            to = reports(i, 2);
            
            % Find a large value to reduce by 1
            [~, max_idx] = max(rounded_values(from:to));
            max_idx = max_idx + from - 1; % Convert to global index
            
            if max_idx ~= 1705 % Make sure we're not reducing the one we just fixed
                rounded_values(max_idx) = rounded_values(max_idx) - rounded_values(1705);
                fprintf('Balanced by adjusting index %d\n', max_idx);
            else
                % Find second largest
                temp_values = rounded_values(from:to);
                temp_values(max_idx-from+1) = 0; % Zero out the max
                [~, second_max_idx] = max(temp_values);
                second_max_idx = second_max_idx + from - 1; % Convert to global index
                rounded_values(second_max_idx) = rounded_values(second_max_idx) - rounded_values(1705);
                fprintf('Balanced by adjusting index %d\n', second_max_idx);
            end
            break;
        end
    end
end

%% Plot comparison of original vs rounded
figure;
subplot(2,1,1);
plot(daily_values, 'b-', 'LineWidth', 1);
hold on;
plot(rounded_values, 'r-', 'LineWidth', 0.5);
legend('Original', 'Rounded');
title('Comparison of Original vs Rounded Daily Values');
xlabel('Day Index');
ylabel('Value');
grid on;

% Plot absolute differences
subplot(2,1,2);
bar(abs_diffs);
title('Absolute Differences (|Rounded - Original|)');
xlabel('Day Index');
ylabel('Absolute Difference');
grid on;

% Plot relative differences (limited to reasonable range)
figure;
rel_diffs_plot = min(rel_diffs, 200); % Cap at 200% for better visualization
plot(rel_diffs_plot, 'r-');
title('Relative Differences (%) - Capped at 200%');
xlabel('Day Index');
ylabel('Relative Difference (%)');
grid on;
hold on;
% Add a reference line at 50%
plot([1, length(daily_values)], [50, 50], 'k--');
legend('Relative Difference', '50% Threshold');

% Create detailed view around problematic indices
figure;
window = 20; % Show 20 days before and after
start_idx = max(1, 1704 - window);
end_idx = min(length(daily_values), 1705 + window);

subplot(2,1,1);
plot(start_idx:end_idx, daily_values(start_idx:end_idx), 'b-', 'LineWidth', 1.5);
hold on;
plot(start_idx:end_idx, rounded_values(start_idx:end_idx), 'r-', 'LineWidth', 1);
title(sprintf('Detailed View Around Indices 1704-1705 (±%d days)', window));
xlabel('Day Index');
ylabel('Value');
legend('Original', 'Rounded');
grid on;

% Mark indices 1704 and 1705
plot([1704, 1704], ylim, 'g--');
plot([1705, 1705], ylim, 'g--');
text(1704, max(ylim)*0.9, '1704', 'Color', 'g', 'FontWeight', 'bold');
text(1705, max(ylim)*0.8, '1705', 'Color', 'g', 'FontWeight', 'bold');

% Plot abs differences in this region
subplot(2,1,2);
bar(start_idx:end_idx, abs_diffs(start_idx:end_idx));
title('Absolute Differences in This Region');
xlabel('Day Index');
ylabel('Absolute Difference');
grid on;

% Mark indices 1704 and 1705
hold on;
plot([1704, 1704], ylim, 'g--');
plot([1705, 1705], ylim, 'g--');
text(1704, max(ylim)*0.9, '1704', 'Color', 'g', 'FontWeight', 'bold');
text(1705, max(ylim)*0.8, '1705', 'Color', 'g', 'FontWeight', 'bold');

%% 9. Save results
% Save both the original reconstructed values and the rounded values
final_results = struct();
final_results.original_daily = daily_values;
final_results.rounded_daily = rounded_values;
final_results.monthly_reports = reports;
final_results.parameters = struct('lambda', lambda, 'alpha', alpha, 'gama', gama, 'ratio', ratio);

save('daily_reconstruction_results8.mat', 'final_results', 'Out', 'Out_A1', 'Out_AL');
fprintf('\nResults saved to daily_reconstruction_results8.mat\n');

%% 10. Visualize results
figure;
% Plot reconstructed daily values
subplot(2,1,1);
plot(daily_values);
hold on;
plot(rounded_values, 'r-');
title('Reconstructed Daily Values');
xlabel('Day (Jan 2020 - Nov 2024)');
ylabel('Daily Value');
legend('Original Reconstruction', 'Rounded (Sum-Preserving)');
grid on;

% Plot difference between original and rounded
subplot(2,1,2);
plot(daily_values - rounded_values);
title('Difference: Original - Rounded');
xlabel('Day (Jan 2020 - Nov 2024)');
ylabel('Difference');
grid on;

% Plot monthly comparison
figure;
months = 1:size(reports, 1);
monthly_avg_original = zeros(size(months));
monthly_avg_rounded = zeros(size(months));

for i = months
    from = reports(i, 1);
    to = reports(i, 2);
    days_in_month = to - from + 1;
    
    monthly_avg_original(i) = sum(daily_values(from:to)) / days_in_month;
    monthly_avg_rounded(i) = sum(rounded_values(from:to)) / days_in_month;
end

bar([monthly_avg_original', monthly_avg_rounded']);
title('Monthly Average Daily Values');
xlabel('Month (1 = Jan 2020)');
ylabel('Average Daily Value');
legend('Original Reconstruction', 'Rounded (Sum-Preserving)');
grid on;

%% Helper function for formatted output
function result = conditional(condition, true_value, false_value)
    if condition
        result = true_value;
    else
        result = false_value;
    end
end

%% %% 10. Visualize results
figure;

% Plot reconstructed daily values
subplot(2,1,1);
plot(daily_values, 'b');
hold on;
plot(rounded_values, 'r');
legend('Original Reconstruction', 'Rounded');
xlabel('Day Index');
ylabel('Value');
title('Daily Reconstructed vs Rounded Values');
grid on;

% Plot monthly sums comparison
subplot(2,1,2);
monthly_indices = 1:size(reports, 1);
original_sums = reports(:, 3);
rounded_sums = arrayfun(@(i) sum(rounded_values(reports(i,1):reports(i,2))), monthly_indices');

bar(monthly_indices, [original_sums, rounded_sums]);
legend('Original Monthly', 'Rounded Monthly');
xlabel('Month Index');
ylabel('Monthly Sum');
title('Monthly Sum Comparison (Original vs Rounded)');
grid on;
%% %% Visualisasi: Nilai Harian Setelah Disagregasi dan Pembulatan

figure;
bar(rounded_values, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Hari ke-');
ylabel('Nilai');
title('Nilai Harian Setelah Disagregasi dan Pembulatan');
grid on;
%% %% Visualisasi: Diagram Garis Nilai Harian Setelah Disagregasi dan Pembulatan

figure;
plot(rounded_values, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'Color', [0.2 0.6 0.8]);
xlabel('Hari ke-');
ylabel('Nilai');
title('Nilai Harian Setelah Disagregasi dan Pembulatan');
grid on;
%% 
monthly_values = reports(:, 3);
% Subplot 1: Grafik bulanan (sebelum disagregasi)
subplot(2,1,1); % 2 baris, 1 kolom, grafik pertama
plot(monthly_values, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', [0.1 0.5 0.7]);
title('Sebelum Disagregasi (Bulanan)');
xlabel('Bulan ke-');
ylabel('Nilai');
grid on;

% Subplot 2: Grafik harian (setelah disagregasi)
subplot(2,1,2); % Grafik kedua
plot(rounded_values, '-o', 'LineWidth', 1.2, 'MarkerSize', 3, 'Color', [0.9 0.4 0.2]);
title('Setelah Disagregasi (Harian)');
xlabel('Hari ke-');
ylabel('Nilai');
grid on;


%% Export daily values to Excel with datetime and final rounded values
function export_daily_to_excel(daily_values, rounded_values, filename)
    % Check if we have the same number of values
    if length(daily_values) ~= length(rounded_values)
        error('Original and rounded values must have the same length');
    end
    
    % Create a start date (assuming the data starts on January 1, 2020)
    % You can change this to your actual start date
    start_date = datetime(2020, 1, 1);
    
    % Create dates for each daily value
    dates = start_date + days(0:(length(daily_values)-1));
    
    % Create a table with the data
    T = table(dates', daily_values, rounded_values, ...
        'VariableNames', {'Date', 'Original_Value', 'Rounded_Value'});
    
    % Add absolute and relative differences
    T.Absolute_Difference = abs(T.Rounded_Value - T.Original_Value);
    
    % For relative difference, handle possible division by zero
    rel_diff = zeros(size(daily_values));
    for i = 1:length(daily_values)
        if daily_values(i) > 0
            rel_diff(i) = 100 * abs(rounded_values(i) - daily_values(i)) / daily_values(i);
        else
            rel_diff(i) = 0;
        end
    end
    
    T.Relative_Difference_Percent = rel_diff;
    
    % Write the table to Excel
    writetable(T, filename);
    
    fprintf('Data successfully exported to %s\n', filename);
    
    % Additional statistics to display
    fprintf('Export statistics:\n');
    fprintf('Total days exported: %d\n', height(T));
    fprintf('Date range: %s to %s\n', datestr(T.Date(1)), datestr(T.Date(end)));
    fprintf('Number of zero values in rounded data: %d\n', sum(T.Rounded_Value == 0));
    
    % Find problematic indices (zeros or large differences)
    zero_indices = find(T.Rounded_Value == 0);
    large_diff_indices = find(T.Relative_Difference_Percent > 50);
    
    if ~isempty(zero_indices)
        fprintf('Zero values found at indices:\n');
        for i = 1:min(10, length(zero_indices))
            idx = zero_indices(i);
            fprintf('Index %d (%s): Original=%.4f, Rounded=%d\n', ...
                idx, datestr(T.Date(idx)), T.Original_Value(idx), T.Rounded_Value(idx));
        end
        if length(zero_indices) > 10
            fprintf('... and %d more\n', length(zero_indices) - 10);
        end
    end
    
    if ~isempty(large_diff_indices)
        fprintf('Large differences (>50%%) found at indices:\n');
        for i = 1:min(10, length(large_diff_indices))
            idx = large_diff_indices(i);
            fprintf('Index %d (%s): Original=%.4f, Rounded=%d, Diff=%.2f%%\n', ...
                idx, datestr(T.Date(idx)), T.Original_Value(idx), T.Rounded_Value(idx), ...
                T.Relative_Difference_Percent(idx));
        end
        if length(large_diff_indices) > 10
            fprintf('... and %d more\n', length(large_diff_indices) - 10);
        end
    end
end

%% % After completing the rounding process and having daily_values and rounded_values:
% Export to Excel
export_daily_to_excel(daily_values, rounded_values, 'daily_values_report.xlsx');
