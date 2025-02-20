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

figure
n_nodes = length(node_coords);
houses = node_coords(pole_count+1:n_nodes,:);
plot(houses(:,1),houses(:,2), 'o');

%%
n_iterations = 10000;
n_houses = length(houses(:,1));
marked = zeros(n_houses);
lines = [];

for i =1:n_iterations
    % Choose two random nodes
    c_1 = randi(n_houses);
    c_1 = node_coords(c_1, :);

    c_2 = randi(n_houses);
    c_2 = node_coords(c_2, :);

    if norm(c_2 - c_1) < 7
    
        % Create line out of them
        [m, b] = compute_line(c_1, c_2);
    
        % Count how many nodes it approximately fits
        count = 0;
        marked_temp = zeros(n_houses);
        for j = 1:n_houses
            if marked(j) == 0 
                y_calc = m*node_coords(j,1) + b;
                if abs(y_calc - node_coords(j,2)) < 3
                    marked_temp(j) = 1;
                    count = count + 1;
                end
            end
        end
    
        if count > 8
            marked = marked + marked_temp;
            lines = [lines; m, b];
        end
    end

end

% Plot lines found
x = -30:0.01:30;
hold on
for i = 1:length(lines)
    plot(x, lines(i,1)*x + lines(i,2));
end








%%
function [m, b] = compute_line(node_1, node_2)
    m = (node_2(2) - node_1(2)) / (node_2(1) - node_1(1)); % Slope
    b = node_1(2) - m * node_1(1);           % Y-intercept
end