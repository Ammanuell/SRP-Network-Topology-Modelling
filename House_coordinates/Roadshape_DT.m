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
% plot(houses(:,1),houses(:,2), 'o');

DT = delaunay(houses(:,1),houses(:,2));
triplot(DT,houses(:,1),houses(:,2));
