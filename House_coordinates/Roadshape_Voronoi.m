clc; clear; close all;

[edges, node_coords, pole_count] = generate_network(40);
G = graph(edges(:,1), edges(:,2));

% Extract x and y coordinates for plotting
x_coords = node_coords(:, 1);
y_coords = node_coords(:, 2);

% Plot the graph with correct coordinates
p = plot(G);
p.XData = x_coords;
p.YData = y_coords;
p.NodeLabel = {};
axis equal;


n_nodes = length(node_coords);
houses = node_coords(pole_count+1:n_nodes,:);
% figure
%plot(houses(:,1),houses(:,2), 'o');

%%
% Create Voronoi diagram
% voronoi(houses(:,1),houses(:,2));
[vx,vy] = voronoi(houses(:,1),houses(:,2));


% Filter 
% Removing excess edges

filtered_vx = [];
filtered_vy = [];

% First threshold edges by length
for i = 1:length(vx(1,:))
    edge_length = sqrt((vx(1,i)-vx(2,i))^2 + (vy(1,i)-vy(2,i))^2);

    if edge_length <= 5
        filtered_vx = [filtered_vx, vx(:,i)];
        filtered_vy = [filtered_vy, vy(:,i)];
    end
end

% Remove edges that have no houses around it (using midpoint)
% can check if endpoint if vertex only appears once
filtered_vx_2 = [];
filtered_vy_2 = [];

for i = 1:length(filtered_vx(1,:))

    node_1 = [filtered_vx(1,i), filtered_vy(1,i)];
    node_2 = [filtered_vx(2,i), filtered_vy(2,i)];

    distances_1 = vecnorm(houses - node_1, 2, 2); 
    min_dist_1 = min(distances_1);

    distances_2 = vecnorm(houses - node_2, 2, 2); 
    min_dist_2 = min(distances_1);


    if min_dist_1 <= 4 && min_dist_2 <= 4
        filtered_vx_2 = [filtered_vx_2, filtered_vx(:,i)];
        filtered_vy_2 = [filtered_vy_2, filtered_vy(:,i)];
    end
end

% Filter those out of alpha shape
shp = alphaShape(houses(:,1),houses(:,2));
shp.Alpha = 3.9;


filtered_vx_3 = [];
filtered_vy_3 = [];

for i = 1:length(filtered_vx_2(1,:))


    if inShape(shp, filtered_vx_2(1,i), filtered_vy_2(1,i)) && inShape(shp, filtered_vx_2(2,i), filtered_vy_2(2,i))
        filtered_vx_3 = [filtered_vx_3, filtered_vx_2(:,i)];
        filtered_vy_3 = [filtered_vy_3, filtered_vy_2(:,i)];
    end
end


% Look for cycles and remove
% First convert to a graph

% Initialize empty arrays for nodes and edges
node_coords_2 = zeros(0, 2);
edges_2 = [];

% Process each edge
for i = 1:length(filtered_vx_3) - 1
    % Get the two endpoints of the edge
    p1 = [filtered_vx_3(1, i), filtered_vy_3(1, i)];
    p2 = [filtered_vx_3(2, i), filtered_vy_3(2, i)];
    
    % Find or add p1 to node_coords
    [~, idx1] = ismember(p1, node_coords_2, 'rows');
    if idx1 == 0
        node_coords_2 = [node_coords_2; p1];
        idx1 = size(node_coords_2, 1); % New index
    end
    
    % Find or add p2 to node_coords_2
    [~, idx2] = ismember(p2, node_coords_2, 'rows');
    if idx2 == 0
        node_coords_2 = [node_coords_2; p2];
        idx2 = size(node_coords_2, 1); % New index
    end
    
    % Add the edge
    edges_2 = [edges_2; idx1, idx2];
end


% Remove edges that are too small by combining endpoints

% Iterate through edges and merge nodes if the edge is too small
for i = size(edges_2, 1):-1:1 % Iterate backwards to avoid indexing issues
    % Get the indices of the edge endpoints
    node_1_id = edges_2(i, 1);
    node_2_id = edges_2(i, 2);
    
    % Get the coordinates of the two endpoints
    node_1 = node_coords_2(node_1_id, :);
    node_2 = node_coords_2(node_2_id, :);
    
    % Calculate the edge length
    edge_length = norm(node_1 - node_2);
    
    % If the edge is too small, merge the nodes
    if edge_length < 1.45
        % Merge nodes: arbitrarily choose one node to keep
        % Here we choose to keep node_1 and update all references to node_2
        node_coords_2(node_1_id, :) = (node_1 + node_2) / 2; % Update node_1 to the midpoint
        edges_2(edges_2 == node_2_id) = node_1_id; % Redirect edges pointing to node_2 to node_1
        
        % Remove node
        node_coords_2(node_2_id, :) = NaN; % Mark as removed
    end
end

% Remove NaN entries from node_coords_2
node_coords_2 = node_coords_2(~any(isnan(node_coords_2), 2), :);

% Reindex the edges to account for removed nodes
[~, ~, new_id] = unique(edges_2(:));
edges_2 = reshape(new_id, size(edges_2));
% Remove self loops
self_loops = edges_2(:, 1) == edges_2(:, 2);

% Remove these rows from the edges array
edges_2 = edges_2(~self_loops, :);



% Plot Result
hold on
figure
plot(filtered_vx_3, filtered_vy_3);
hold on
plot(houses(:,1),houses(:,2), 'o');

figure
G_2 = graph(edges_2(:,1), edges_2(:,2));

% Extract x and y coordinates for plotting
x_coords_2 = node_coords_2(:, 1);
y_coords_2 = node_coords_2(:, 2);

% Plot the graph with correct coordinates
p = plot(G_2);
p.XData = x_coords_2;
p.YData = y_coords_2;
p.NodeLabel = {};
axis equal;
hold on
plot(houses(:,1),houses(:,2), 'o');

% Remove some nodes(nodes not closest to customer)
marked = zeros(1,length(node_coords_2));
% Mark nodes
for i=1:length(houses)
    house_coord = houses(i,:);
    distances = vecnorm(node_coords_2 - house_coord, 2, 2); 
    [minDist, close_node_index] = min(distances);
    marked(close_node_index) = 1;
end
% Remove unmarked nodes

remove = [];
for i = 1:length(marked)
    if marked(i) == 0
        remove = [remove, i];
    end
end
G_2 = rmnode(G_2, remove);
node_coords_2 = node_coords_2(logical(marked),:);
x_coords_2 = x_coords_2(logical(marked),1);
y_coords_2 = y_coords_2(logical(marked),1);

p = plot(G_2);
p.XData = x_coords_2;
p.YData = y_coords_2;
p.NodeLabel = {};
axis equal;
hold on
plot(houses(:,1),houses(:,2), 'o');

% % Remove cycles
% cycles = cyclebasis(G_2);
% while length(cycles) > 0
%     cycle = cycles{length(cycles)};
%     scores = [];
%     cycle_edges = [];
%     % Calculate scores for each edge
%     for i = 1:length(cycle)
%         % Get edge
%         if i == length(cycle)
%             edge = [cycle(i), cycle(1)];
%             edge = [min(edge),max(edge)]; % Reorder
%         else
%             edge = [cycle(i),cycle(i+1)];
%             edge = [min(edge),max(edge)]; % Reorder
%         end
%         cycle_edges = [cycle_edges; edge];
%         scores = [scores, calculate_score(G_2, edge, node_coords_2)];
%     end
% 
%     % cycle_edges
%     % scores
%     % Remove max similarity score
%     [~, index] = max(scores);
%     G_2 = rmedge(G_2, cycle_edges(index, 1), cycle_edges(index, 2));
%     scores(index) = 0;
%     cycle_edges(index,:);
%     while sum(scores>50) > 0
%         % scores
%         % cycle_edges(index,:)
%         [~, index] = max(scores);
%         G_2 = rmedge(G_2, cycle_edges(index, 1), cycle_edges(index, 2));
%         scores(index) = 0;
%     end
%     G_2 = rmedge(G_2, cycle_edges(index, 1), cycle_edges(index, 2));
% 
%     % Update cycles
%     cycles = cyclebasis(G_2);
% end
% 
% figure
% 
% % Extract x and y coordinates for plotting
% x_coords_2 = node_coords_2(:, 1);
% y_coords_2 = node_coords_2(:, 2);
% 
% % Plot the graph with correct coordinates
% p = plot(G_2);
% p.XData = x_coords_2;
% p.YData = y_coords_2;
% p.NodeLabel = {};
% axis equal;
% hold on
% plot(houses(:,1),houses(:,2), 'o');

% Create graph with weights

edges_2 = table2array(G_2.Edges);
weights = zeros(length(edges_2), 1);

for i = 1:length(weights)
    weights(i) = calculate_score(G_2, edges_2(i,:), node_coords_2);
end

G_3 = graph(edges_2(:,1), edges_2(:,2), weights);
G_3 = minspantree(G_3);

figure;
p = plot(G_3);
p.XData = x_coords_2;
p.YData = y_coords_2;
p.NodeLabel = {};
axis equal;
hold on
plot(houses(:,1),houses(:,2), 'o');

%%
% edges_2 = table2array(G_2.Edges);
% 
% scores = [];
% for i = 1:length(edges_2)
%     score = calculate_score(G_2, edges_2(i,:), node_coords_2);
%     scores = [scores, score];
% end

%%
get_connected_edges(G_2,[20,32]);
calculate_score(G_2, [26,27], node_coords_2);


function score = calculate_score(G, edge, node_coords)
    connected_edges = get_connected_edges(G,edge);


    angle_diffs = [];
    edge_angle = get_angle(node_coords(edge(1), :), node_coords(edge(2), :));
    for k = 1:length(connected_edges(:,1))
        node_1 = node_coords(connected_edges(k,1),:);
        node_2 = node_coords(connected_edges(k,2),:);
        angle_diff = min(abs( edge_angle - get_angle(node_1, node_2) ),abs( edge_angle - get_angle(node_2, node_1) )) ;
        angle_diffs = [angle_diffs, angle_diff];
    end
    score = min(angle_diffs);
end

function connected_edges = get_connected_edges(G,edge)
    connected_edges = [];

    edges = table2array(G.Edges);

    for i = 1:length(edges)
        if (ismember(edges(i,1), edge) || ismember(edges(i,2), edge)) && any(edges(i,:) ~= edge)
           connected_edges = [connected_edges ; edges(i,:)];
        end
    end
end

function angle = get_angle(node_1, node_2)
    angle = atan2d(node_2(2) - node_1(2), node_2(1) - node_1(1));
end

