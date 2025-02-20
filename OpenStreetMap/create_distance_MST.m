function [G, pole_positions] = create_distance_MST(houses_longlat, pole_positions, randomness)

    
    % Create edges
    edges = [];
    weights = [];

    lower_bound = 1 - randomness;
    a = lower_bound;
    b = 2*(1-lower_bound);
    
    % Create edges between house and closest pole
    n = length(pole_positions);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            rng = a + b*rand();
            % dist = rng*abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        % Create edge
        edges = [edges; [n+i, closest_pole]];
        weights = [weights; min_dist];
    end


    % Edges between pole_positions, weighted based on distance
    edges = [edges ; nchoosek(1:length(pole_positions), 2)];
  
    
    for i= length(houses_longlat) + 1:length(edges)
        edge = edges(i,:);
        
        pole_1 = pole_positions(edge(1),:);
        pole_2 = pole_positions(edge(2),:);
        
        rng = a + b*rand();
        dist = rng * abs(haversine(pole_1, pole_2));
        
        weights = [weights; dist];
    end
    
    
    % Create graph and MST
    G = graph(edges(:,1), edges(:,2), weights);
    G = minspantree(G);

end

function d = haversine(point_1, point_2)
    R = 6378137;
    
    lon1 = deg2rad(point_1(1));
    lat1 = deg2rad(point_1(2));

    lon2 = deg2rad(point_2(1));
    lat2 = deg2rad(point_2(2));

    delta_lat = lat2 - lat1;
    delta_lon = lon2 - lon1;

    % Haversine formula
    a = sin(delta_lat/2)^2 + cos(lat1) * cos(lat2) * sin(delta_lon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1 - a));

    % Compute distance
    d = R * c/1000;
end

