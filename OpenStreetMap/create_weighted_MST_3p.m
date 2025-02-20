function [G, pole_positions] = create_weighted_MST_3p(houses_longlat, pole_positions, house_voltages)

    
    % Create edges
    edges = [];
    weights = [];
    
    % Remove poles not closest to a house
    n = length(pole_positions);
    poles_connected = zeros(1, n);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        poles_connected(closest_pole) = 1;
        
    end
    pole_positions = pole_positions(poles_connected~=0, :);

    % Connect each customer to closest pole
    n = length(houses_longlat);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        % Create edge
        edges = [edges; [i, closest_pole+n]];
        weights = [weights; min_dist];
        
    end
    
    G_temp = graph(edges(:,1), edges(:,2), weights);

    % figure
    % p = plot(G_temp);
    % p.XData = [houses_longlat(:,1); pole_positions(:,1)];
    % p.YData = [houses_longlat(:,2); pole_positions(:,2)];

    % Calculate 'voltage' of each pole
    % pole_voltages = zeros(length(pole_positions), length(house_voltages(1,:)));
    pole_voltages = zeros(length(pole_positions), 2000, 3);

    % For all poles connected to houses, voltage is average of them
    for i = n+1:n+length(pole_positions)
        houses = neighbors(G_temp, i);
        %houses = houses(houses>length(pole_positions)); % Get all houses connected to pole
        
        if ~isempty(houses)
            pole_voltages(i-n,:) = mean(house_voltages(houses, :),1, 'omitnan');
        end
    end



    % Create edges between pole, weighted based on voltage and distance
    start = length(edges);
    edges = [edges ; nchoosek(n+1:n+length(pole_positions), 2)];
  
    corr_table_A = corrcoef(pole_voltages(:,:,1)');
    %corr_table_A(isnan(corr_table_A))= 0;
    
    corr_table_B = corrcoef(pole_voltages(:,:,2)');
    %corr_table_B(isnan(corr_table_B))= 0;
    
    corr_table_C = corrcoef(pole_voltages(:,:,3)');
    %corr_table_C(isnan(corr_table_C))= 0;

    dists = [];
    corrs = [];
    for i = start+1:length(edges)
        edge = edges(i,:);
        
        pole_1 = pole_positions(edge(1)-n,:);
        pole_2 = pole_positions(edge(2)-n,:);
        
        dist = abs(haversine(pole_1, pole_2));

        voltage_diff = abs(pole_voltages(edge(1)-n) - pole_voltages(edge(2)-n));

        % Take average correlation (omitting 0/NaN corr vlaues)
        % corr =  corr_table_A(edge(1), edge(2)) + corr_table_B(edge(1), edge(2)) + corr_table_C(edge(1), edge(2));
        corr = mean([corr_table_A(edge(1)-n, edge(2)-n), corr_table_B(edge(1)-n, edge(2)-n), corr_table_C(edge(1)-n, edge(2)-n)], 'omitnan');
        if isnan(corr)
            corr = 0;
        end
        
        dists = [dists ;  dist];
        corrs = [corrs ; corr];
        % weight = dist + 0.0001*voltage_diff;
        % weight = alpha*corr
        %weight = dist - 0.005*corr;

        
        %weights = [weights; weight];
    end
    % normalise distances and correlations
    dists = (dists-min(dists))/(max(dists) - min(dists));
    corrs = (corrs-min(corrs))/(max(corrs) - min(corrs));
    corrs = (corrs-0.97)/(max(corrs) - min(corrs));

    new_weights = dists - 0.1*corrs;
    weights = [weights ; new_weights];
    
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

