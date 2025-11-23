% Two-stage grid search with automatic finer-grid refinement
clc; clear; close all;
fprintf('Lorenz attractor: coarse grid + automatic finer-grid refinement\n');

TT = 1000;        % Transient time
CT = 10;          % Computation time
WT = 0.15;        % Window time
h = 0.001;        % Integration step time
y = 20;           % Decimation coeff
iters = 200;

% Coarse grid
Kf_start = -100; Kf_end = 100; Kf_step = 5;
Kb_start = -100; Kb_end = 100; Kb_step = 5;

% Refinement
do_refine = false;
refine_factor = 10;
refine_radius_steps = 1;

% Success = log10(RMS_end / RMS_start) < -1
threshold = -1;

IC_m = [3 -3 0];
IC_s = [0 0 0];

params.sigma = 10;
params.rho  = 28;
params.beta = 8/3;

% Parallel pool
if isempty(gcp('nocreate'))
    parpool('local');
end
pool = gcp();
fprintf('Parallel computing (%d workers)\n', pool.NumWorkers);

% Transient
TT_points = ceil(TT/h);
X_m = IC_m;
for i = 1:TT_points
    X_m = lorenz_step_cd(params, h, X_m);
end
IC_m = X_m;

% Data generation
CT_points_y = ceil(CT/h/y);
XYZ_arr = zeros(CT_points_y, 3);

CT_points = ceil(CT/h);
ptr = 0;
for i = 1:CT_points
    X_m = lorenz_step_cd(params, h, X_m);
    if mod(i, y) == 0
        ptr = ptr + 1;
        XYZ_arr(ptr, :) = X_m;
    end
end

% Coarse grid
WT_points = ceil(WT/h);

Kf_vals = Kf_start:Kf_step:Kf_end;
Kb_vals = Kb_start:Kb_step:Kb_end;

fprintf('Coarse grid:\n');
fprintf('Kf = [%g ... %g], step = %g, n = %d\n', ...
    Kf_start, Kf_end, Kf_step, numel(Kf_vals));
fprintf('Kb = [%g ... %g], step = %g, n = %d\n', ...
    Kb_start, Kb_end, Kb_step, numel(Kb_vals));

[success_map, coarse_time] = create_map(Kf_vals, Kb_vals, XYZ_arr,...
    params, WT_points, h, iters, IC_s, threshold);

fprintf('Coarse grid done in %d seconds\n', round(coarse_time));

% Best point on coarse grid
[best_val, idx_val] = max(success_map(:));
[i_bestKb, i_best_Kf] = ind2sub(size(success_map), idx_val);
best_Kf = Kf_vals(i_best_Kf);
best_Kb = Kb_vals(i_bestKb);

fprintf('Maximum on coarse grid: Kf = %g, Kb = %g, best percent = %.2f\n',...
    best_Kf, best_Kb, best_val);

% Plot coarse grid
figure('Position', [250, 50, 1000, 730]);
imagesc(Kf_vals, Kb_vals, success_map);  % X=Kf, Y=Kb
axis xy;
xlabel('$K_{forw}$', 'Interpreter','latex','FontSize',14);
ylabel('$K_{back}$', 'Interpreter','latex','FontSize',14);
title(sprintf(['Percent of points synchronized by criterion ' ...
    '$\\Delta log_{10}(RMS(||Error||)) < %d$'], threshold),...
    'Interpreter','latex','FontSize',14);
colormap(turbo);
cb = colorbar;
set(cb.Label,'String','$percent$','Interpreter','latex','FontSize',14);
caxis([0, max(success_map(:))]);
hold on;
plot(best_Kf, best_Kb,'kp','MarkerSize',10,'MarkerFaceColor','w');
text(best_Kf, best_Kb, sprintf('   (%.1f, %.1f)', best_Kf, best_Kb));


% ---------------- REFINEMENT ----------------
if do_refine
    fprintf('Starting refinement...\n');

    Kf_fine_step = Kf_step / refine_factor;
    Kb_fine_step = Kb_step / refine_factor;
    r = refine_radius_steps;

    Kf_vals_fine = (best_Kf - r*Kf_step) : Kf_fine_step : (best_Kf + r*Kf_step);
    Kb_vals_fine = (best_Kb - r*Kb_step) : Kb_fine_step : (best_Kb + r*Kb_step);

    fprintf('Refine grid:\n');
    fprintf('Kf = [%g ... %g], step = %g, n = %d\n', ...
        Kf_vals_fine(1), Kf_vals_fine(end), Kf_fine_step, numel(Kf_vals_fine));
    fprintf('Kb = [%g ... %g], step = %g, n = %d\n', ...
        Kb_vals_fine(1), Kb_vals_fine(end), Kb_fine_step, numel(Kb_vals_fine));

    [success_map_fine, fine_time] = create_map(...
        Kf_vals_fine, Kb_vals_fine, XYZ_arr, params,...
        WT_points, h, iters, IC_s, threshold);

    fprintf('Refinement grid done in %d seconds\n', round(fine_time));

    [best_val_ref, idx_val_ref] = max(success_map_fine(:));
    [i_bestKb_ref, i_bestKf_ref] = ind2sub(size(success_map_fine), idx_val_ref);
    best_Kf_ref = Kf_vals_fine(i_bestKf_ref);
    best_Kb_ref = Kb_vals_fine(i_bestKb_ref);

    fprintf('Refined best: Kf = %g, Kb = %g, percent = %.2f\n',...
        best_Kf_ref, best_Kb_ref, best_val_ref);

    figure('Position',[250,50,1000,730]);
    imagesc(Kf_vals_fine, Kb_vals_fine, success_map_fine);
    axis xy;
    xlabel('$K_{forw}$','Interpreter','latex','FontSize',14);
    ylabel('$K_{back}$','Interpreter','latex','FontSize',14);
    title(sprintf(['Percent of points synchronized by criterion ' ...
        '$\\Delta log_{10}(RMS(||Error||)) < %d$'], threshold),...
        'Interpreter', 'latex', 'FontSize', 14);
    colormap(turbo);
    cb = colorbar;
    set(cb.Label,'String', '$percent$', 'Interpreter', 'latex', 'FontSize', 14);
    caxis([0, max(success_map_fine(:))]);
    hold on;

    % Отметка лучшей точки
    plot(best_Kf_ref, best_Kb_ref, 'kp', 'MarkerSize',10, 'MarkerFaceColor', 'w');
    text(best_Kf_ref, best_Kb_ref, sprintf('   (%.2f, %.2f)', best_Kf_ref, best_Kb_ref));
end


%% ---------------- FUNCTIONS ----------------

function [success_map, elapsed] = create_map(Kf_vals, Kb_vals,...
    XYZ_arr, params, WT_points, h, iters, IC_s, threshold)

    start_time = tic;
    nKf = numel(Kf_vals);
    nKb = numel(Kb_vals);
    CT_points_y = size(XYZ_arr, 1);
    success_map = zeros(nKb, nKf);  % rows = Kb, cols = Kf

    for iKb = 1:nKb
        for iKf = 1:nKf

            Kf = Kf_vals(iKf);
            Kb = Kb_vals(iKb);

            log_rms_error = zeros(1, CT_points_y);

            parfor iPoint = 1:CT_points_y
                X_m = XYZ_arr(iPoint, :);
                X_s = IC_s;

                WT_forw = zeros(WT_points, 3);
                for t = 1:WT_points
                    WT_forw(t,:) = X_m;
                    X_m = lorenz_step_cd(params, h, X_m);
                end
                WT_back = flip(WT_forw, 1);

                buffer_rms = zeros(1, iters);
                buffer_norm = zeros(1, WT_points - 1);

                K_forw = [0 Kf 0];
                K_back = [0 Kb 0];

                for it = 1:iters
                    for j = 1:WT_points - 1
                        buffer_norm(j) = norm(WT_forw(j,:) - X_s);
                        X_s = sync_pc(params, h, K_forw, WT_forw(j,:), X_s);
                    end
                    for j = 1:WT_points - 1
                        X_s = sync_pc(params, -h, -K_back, WT_back(j,:), X_s);
                    end
                    buffer_rms(it) = rms(buffer_norm);
                end

                if buffer_rms(1) <= 0 || buffer_rms(end) <= 0
                    log_rms_error(iPoint) = 1e6;
                else
                    log_rms_error(iPoint) = log10(buffer_rms(end)) - log10(buffer_rms(1));
                end
            end

            success_map(iKb, iKf) = sum(log_rms_error < threshold) / numel(log_rms_error) * 100;
        end
    end

    elapsed = toc(start_time);
end


function X = sync_pc(params, h, K, X_m, X_s)
    U = K .* (X_m - X_s);
    X = lorenz_step_cd(params, h, X_s);
    X = X + U * h;
end


function X = lorenz_step_cd(params, h, X)
    s  = 0.5;
    h1 = h * s;
    h2 = h * (1 - s);

    X(1) = X(1) + h1 * (params.sigma * (X(2) - X(1)));
    X(2) = X(2) + h1 * (X(1)*(params.rho - X(3)) - X(2));
    X(3) = X(3) + h1 * (X(1)*X(2) - params.beta*X(3));

    X(3) = (X(3) + h2*(X(1)*X(2))) / (1 + h2*params.beta);
    X(2) = (X(2) + h2*(X(1)*(params.rho - X(3)))) / (1 + h2);
    X(1) = (X(1) + h2*(params.sigma*X(2))) / (1 + params.sigma*h2);
end
