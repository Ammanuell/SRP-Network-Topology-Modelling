% Code used to study charecteristics of model

% Groundtruth model stays the same, taken from "groundtruth.mat"
% This is unlike generateTopology_OpenStreetMap.m which taks the data from
% openstreet map and generates a new groundtruth topology each time.

clc; clear all; close all;

load('groundtruth.mat');

%% Stats
% Calculate pole voltages
% For all poles connected to houses, voltage is average of them
pole_voltages = zeros(length(pole_positions_groundtruth), 2000, 3);
n = length(pole_positions_groundtruth);
for i = 1:length(pole_positions_groundtruth)
    houses = neighbors(G_groundtruth, i);
    houses = houses(houses>n); % Get all houses connected to pole
    
    if ~isempty(houses)
        pole_voltages(i,:) = mean(house_voltages_3p(houses-n, :),1, 'omitnan');
    end
end

% figure
% heatmap(customer_data.scriptvar.ABC_Nodes{1}, customer_data.scriptvar.ABC_Nodes{1}, corrcoef(customer_data.scriptvar.V1'))
%scatter(distances(G_groundtruth, customer_data.scriptvar.ABC_Nodes{1}, customer_data.scriptvar.ABC_Nodes{1}), corrcoef(customer_data.scriptvar.V1'))

house_corr = corrcoef(house_voltages_1p');
pole_corr_A = corrcoef(pole_voltages(:,:,1)');
pole_corr_B = corrcoef(pole_voltages(:,:,2)');
pole_corr_C = corrcoef(pole_voltages(:,:,3)');


%%  Plot ground truth & minimum spanning tree
% Ground Truth
figure
set(gcf, 'Position', [100, 100, 1200, 600]);
subplot(1,2,1)
p = plot(G_groundtruth);
p.XData = [pole_positions_groundtruth(:,1);  houses_longlat(:,1)];
p.YData = [pole_positions_groundtruth(:,2); houses_longlat(:,2)];
title('Ground Truth Model')

% Label houses
house_ids = [length(pole_positions_groundtruth)+1:length(pole_positions_groundtruth) + length(houses_longlat)];

house_classifications = strings(length(house_ids),1);
house_classifications(customer_data.scriptvar.ABC_Nodes{1}-length(pole_positions_groundtruth)) = "A";
house_classifications(customer_data.scriptvar.ABC_Nodes{2}-length(pole_positions_groundtruth)) = "B";
house_classifications(customer_data.scriptvar.ABC_Nodes{3}-length(pole_positions_groundtruth)) = "C";

for i=house_ids
    if house_classifications(i-length(pole_positions_groundtruth)) == "A"
        highlight(p, i, 'NodeColor','r')
    end
    if house_classifications(i-length(pole_positions_groundtruth)) == "B"
        highlight(p, i, 'NodeColor','g')
    end
    if house_classifications(i-length(pole_positions_groundtruth)) == "C"
        highlight(p, i, 'NodeColor','b')
    end
end

house_labels = string(round(house_voltages_1p(:,1)));
labelnode(p, house_ids, house_labels);


% Simple Minimimum spanning tree (for comparison) to show how groundtruth isn't ideal
[G_estimate, pole_positions_estimate] = create_distance_MST(houses_longlat, pole_positions_groundtruth, 0);
subplot(1,2,2)
p = plot(G_estimate);
p.XData = [pole_positions_estimate(:,1);  houses_longlat(:,1)];
p.YData = [pole_positions_estimate(:,2); houses_longlat(:,2)];
p.NodeLabel = {};
title('Simple Distance MST')

% Plot estimate as well
figure
[G_estimate, pole_positions_estimate] = create_iterative_tree(houses_longlat, pole_positions_groundtruth, house_voltages_3p);
% [G_estimate, pole_positions_estimate] = create_weighted_MST_3p(houses_longlat, pole_positions_groundtruth, house_voltages_1p);

p = plot(G_estimate);
p.XData = [houses_longlat(:,1); pole_positions_estimate(:,1)];
p.YData = [houses_longlat(:,2); pole_positions_estimate(:,2)];
p.NodeLabel = {};
title('Weighted MST')



