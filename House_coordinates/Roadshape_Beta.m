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
figure
plot(houses(:,1),houses(:,2), 'o');

%%
roads = betaSkeleton(houses, 1.1);

figure
G_2 = graph(roads(:,1), roads(:,2));
cycles = cyclebasis(G_2);

% Extract x and y coordinates for plotting
x_coords_2 = houses(:, 1);
y_coords_2 = houses(:, 2);

% Plot the graph with correct coordinates
p = plot(G_2);
p.XData = x_coords_2;
p.YData = y_coords_2;
p.NodeLabel = {};
axis equal;


% Compute midpoints of edges
midpoints = [];
for i = 1:size(roads, 1)
    node1 = houses(roads(i, 1), :);
    node2 = houses(roads(i, 2), :);
    midpoints = [midpoints; (node1 + node2) / 2]; % Add midpoint
end

%scatter(midpoints(:, 1), midpoints(:, 2), 'red'); % Nodes
% Step 1: Create an alpha shape
alpha = 2.7; % Adjust alpha parameter as needed
shp = alphaShape(midpoints, alpha);

% Step 2: Extract boundary facets (edges on the perimeter)
boundaryEdges = boundaryFacets(shp);

% Step 3: Extract unique perimeter points
boundaryPoints = unique(midpoints(boundaryEdges, :), 'rows');

figure
poles = setdiff(midpoints, boundaryPoints, 'rows');
hold on 
scatter(poles(:,1), poles(:,2), 'r')
scatter(houses(:,1), houses(:,2), 'b')

% Visualization
% figure;
% plot(shp, 'FaceColor', 'none'); % Plot the alpha shape
% hold on;
% scatter(midpoints(:,1), midpoints(:,2), 'r', 'filled'); % Plot all points
% scatter(boundaryPoints(:,1), boundaryPoints(:,2), 'b', 'filled'); % Plot boundary points
% legend('Alpha Shape', 'All Points', 'Boundary Points');
% title('Alpha Shape with Perimeter Points');
% hold off;




%%

function edges = betaSkeleton(node_coords, beta)
    % Function to compute the beta skeleton edges for a set of nodes
    % Input:
    %   node_coords - Nx2 array of node coordinates
    %   beta - scalar value (> 0) controlling the skeleton formation
    % Output:
    %   edges - Mx2 array of edges, where each row contains indices of connected nodes

    % Number of nodes
    num_nodes = size(node_coords, 1);

    % Initialize an empty array for edges
    edges = [];

    % Loop through all pairs of nodes
    for i = 1:num_nodes-1
        for j = i+1:num_nodes
            % Get the coordinates of the two nodes
            p1 = node_coords(i, :);
            p2 = node_coords(j, :);

            % Calculate the midpoint and radius of the circle/lens
            midpoint = (p1 + p2) / 2;
            distance = norm(p1 - p2);
            radius = beta * distance / 2;

            % Define the region to check for emptiness
            if beta <= 1
                % Lens region (intersection of two circles)
                theta = atan2(p2(2) - p1(2), p2(1) - p1(1));
                offset = sqrt((distance / 2)^2 - (beta * distance / 2)^2);
                c1 = midpoint + offset * [-sin(theta), cos(theta)];
                c2 = midpoint - offset * [-sin(theta), cos(theta)];

                % Check if any other nodes are inside either circle
                is_empty = all(arrayfun(@(k) norm(node_coords(k, :) - c1) > radius && ...
                                              norm(node_coords(k, :) - c2) > radius, ...
                                              setdiff(1:num_nodes, [i, j])));
            else
                % Circle region (circle centered at midpoint)
                is_empty = all(arrayfun(@(k) norm(node_coords(k, :) - midpoint) > radius, ...
                                              setdiff(1:num_nodes, [i, j])));
            end

            % If the region is empty, add the edge
            if is_empty
                edges = [edges; i, j];
            end
        end
    end
end
