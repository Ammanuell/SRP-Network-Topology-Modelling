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
function [edges, node_coords, pole_count] = generate_network(n_poles)
    edges = [];
    node_coord = [0, 0];  % origin
    node_coords = node_coord;  % Stores all node coordinates as [x, y]
    pole_count = 1;


    s_houses = 5;
    direction = 0;
    
    [edges, node_coords, pole_count] = branch_random_street(edges, node_coord, 1, node_coords, pole_count, n_poles, s_houses, direction);
    edges = unique(sort(edges, 2), 'rows');
    
    [edges, node_coords] = add_houses(edges, node_coords, 1.4);
end



function [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, r_poles, s_houses, direction, curve)
    % Base case: stop recursion if no remaining poles
    if r_poles <= 0
        return;
    end

    prev_coord = origin_coord;
    prev_no = origin_no;

    for i = 2:min(r_poles,s_houses)

        % Calculate and store coordinates of node
        node_coord = prev_coord + choose_pole_spacing()*[cosd(direction), sind(direction)];

        % Check if too close to another node
        distances = vecnorm(node_coords - node_coord, 2, 2); 
        [minDist, close_node_index] = min(distances);

        if minDist < 1.8
            % No cycles

            break;
        else
             node_coords = [node_coords; node_coord];

            % Connect to the previous node
            edges =[edges; [prev_no, pole_count + 1]];
    
    
            % Update pole count and previous node
            pole_count = pole_count + 1;
            prev_no = pole_count;
            prev_coord = node_coord;
        end

        direction = direction + curve;
               
    end

    remaining_poles = r_poles - s_houses + 1;
    end_no = prev_no;

    % Recursive call for branching
    [edges, node_coords, pole_count] = branch_random_street(edges, prev_coord, end_no, node_coords, pole_count, remaining_poles, s_houses, direction);

end


function [edges, node_coords, pole_count] = branch_random_street(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles, s_houses, direction)
     choice = rand();
     s_houses = 4 + randi(3);

    if choice < 0.15
        % Keep going straight
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no,  node_coords, pole_count, remaining_poles, s_houses, direction, 0);
    elseif choice < 0.25
        % Go left
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles, s_houses, direction + 90, 0);
    elseif choice < 0.3
        % Go left curved
        choices = [10,-10];
        curve = choices(randi(2));
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles, s_houses, direction + 90, curve);
    elseif choice < 0.35
        % Go right
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles, s_houses, direction + 270, 0);
    elseif choice < 0.4
        % Go right curved
        choices = [10,-10];
        curve = choices(randi(2));
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles, s_houses, direction + 270, curve);
    elseif choice < 0.5
        % Split left and right
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + 270, 0);
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + 90, 0);
    elseif choice < 0.65
        % Split left and right with curvature
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + 270, 10);
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + 90, -10);
    elseif choice < 0.8
        % Split into 2 angled
        angle = -90 + 180*rand;
        gap = 45 + 90*rand;
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + angle, 0);
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 2, s_houses, direction + angle+gap, 0);
    else
        % Split left, right and straight
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 3, s_houses, direction, 0);
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 3, s_houses, direction + 90, 0);
        [edges, node_coords, pole_count] = generate_street_rec(edges, origin_coord, origin_no, node_coords, pole_count, remaining_poles / 3, s_houses, direction + 270, 0);
    end
end

function [edges, node_coords] = add_houses(edges, node_coords, house_dist)

    n_poles = length(node_coords);
    max_houses_per_pole = 4;
    n_houses = 1;
    
    % Add houses to each pole
    for i = 1:n_poles
        connected_edges = edges(any(edges == i, 2), :);
        house_directions = find_house_directions(i, connected_edges, node_coords, max_houses_per_pole);

        for j = 1:length(house_directions)
            % Add coord
            house_coord = node_coords(i,:) + house_dist*[cosd(house_directions(j)), sind(house_directions(j))];

            % Check if too close to another node
            distances = vecnorm(node_coords - house_coord, 2, 2); 
            [minDist, close_node_index] = min(distances);

            if minDist > 1.3
                node_coords = [node_coords; house_coord];
    
                % Add edge
                edge = [i, n_poles+n_houses];
                edges = [edges ; edge];
    
                n_houses = n_houses + 1;
            end

        end

    end
end

function directions = find_house_directions(node_id, connected_edges, node_coords, max_houses_per_pole)
    angles = [];
    min_angle_diff = 59;


    for i = 1:length(connected_edges(:,1))
        node_2 = connected_edges(i, connected_edges(i,:) ~= node_id);

        coord_1 = node_coords(node_id, :);
        coord_2 = node_coords(node_2, :);
        vec = coord_2 - coord_1;

        angle = mod(atan2d(vec(2), vec(1)), 360);
        angles = [angles, angle];
    end

    directions = [];

    while length(directions) < max_houses_per_pole - length(connected_edges(:,1))
        direction = rand*360;
        valid = 1;
        for i = 1:length(angles)
            angle_diff = norm(direction - angles(i));
            if angle_diff > 180
                angle_diff = 360 - angle_diff;
            end

            if angle_diff < min_angle_diff
                valid = 0;
            end
        end

        if valid
            directions = [directions, direction];
            angles = [angles, direction];
        end
    end
end

function spacing = choose_pole_spacing()
    choice = rand();

    if choice < 1
        spacing = 1;
    elseif choice < 0.6
        spacing = 2;
    elseif choice < 0.8
        spacing = 3;
    else
        spacing = 4;
    end

    spacing = 2 + 2*rand;
end

