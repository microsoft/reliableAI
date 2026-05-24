function T = AstarKtree(R,M)
% output: T is a matrix representing a k-tree
% input: R is the root clique, M is the mutual information of any pair

% each element in the open or close set is a vector of size k+1
% the first element in the vector is the node n to be added
% the remaining k element is the clique in the current graph
% there is a like between n and each node in the clique in the new graph
% tic
% M = rand(150);
% M = (M + M')/2;
% M = setdiag(M, 0);
% R = [1,2,3,4,5];
k = length(R)-1;

n = size(M, 1);
G = eye(n);
G(R,R) = 1;


closedset = [];
openset = R;
current_path = [];
while ~isempty(openset)
    current_score = 0;
    for thread = 1:size(openset, 1)
        next_step = [current_path; openset(thread, :)];
        tempscore = f_score(next_step, M);
        if current_score > tempscore
            current_score = tempscore;
            current = thread;
            future_path = next_step;
        end
    end
    
    if size(future_path, 1) == n-k
%         toc
        T = buildgraph(future_path, n);
        return;
    end
    
    node_ind = openset(current, 1);
    ind = find(openset(:, 1)==node_ind);
    closedset = [closedset; openset(ind,:)];
    openset(ind, :) = [];
    
    
    
    %% find all the cliques of size k
    G = buildgraph(future_path, n);
%     [components,cliques,CC] = k_clique(k, G);
    C = find_cliques(G,k);
    explored_nodes = [R, future_path(2:end,1)'];
    remain_nodes = setdiff(1:n, explored_nodes);
 
    for thread = 1:size(C, 1)
        for j = remain_nodes
            
            if ismember([j, C(thread, :)], closedset, 'rows')
                continue;
            end
            
            if ~ismember([j, C(thread, :)], openset, 'rows')
                openset = [openset; [j, C(thread, :)]];
            end
        end
    end
    
    current_path = future_path;  
end

            