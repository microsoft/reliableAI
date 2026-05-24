function c = find_cliques(M, k)
nb_nodes = size(M,1); % number of nodes 

% Find the largest possible clique size via the degree sequence:
% Let {d1,d2,...,dk} be the degree sequence of a graph. The largest
% possible clique size of the graph is the maximum value k such that
% dk >= k-1
degree_sequence = sort(sum(M,2) - 1,'descend');
max_s = 0;
for thread = 1:length(degree_sequence)
    if degree_sequence(thread) >= thread - 1
        max_s = thread;
    else 
        break;
    end
end

cliques = cell(0);
% Find all s-size kliques in the graph
for s = max_s:-1:3
    M_aux = M;
    % Looping over nodes
    for n = 1:nb_nodes
        A = n; % Set of nodes all linked to each other
        B = setdiff(find(M_aux(n,:)==1),n); % Set of nodes that are linked to each node in A, but not necessarily to the nodes in B
        C = transfer_nodes(A,B,s,M_aux); % Enlarging A by transferring nodes from B
        if ~isempty(C)
            for thread = size(C,1)
                cliques = [cliques;{C(thread,:)}];
            end
        end
        M_aux(n,:) = 0; % Remove the processed node
        M_aux(:,n) = 0;
    end
end

l = length(cliques);
c = [];
for thread = 1:l
    tmp = cliques{thread};
    if length(tmp)>=k
        c = [c; nchoosek(tmp,k)];
    end
end

    function R = transfer_nodes(S1,S2,clique_size,C)
        % Recursive function to transfer nodes from set B to set A (as
        % defined above)
        
        % Check if the union of S1 and S2 or S1 is inside an already found larger
        % clique
        found_s12 = false;
        found_s1 = false;
        for c = 1:length(cliques)
            for cc = 1:size(cliques{c},1)
                if all(ismember(S1,cliques{c}(cc,:)))
                    found_s1 = true;
                end
                if all(ismember(union(S1,S2),cliques{c}(cc,:)))
                    found_s12 = true;
                    break;
                end
            end
        end
        
        if found_s12 || (length(S1) ~= clique_size && isempty(S2))
            % If the union of the sets A and B can be included in an
            % already found (larger) clique, the recursion is stepped back
            % to check other possibilities
            R = [];
        elseif length(S1) == clique_size;
            % The size of A reaches s, a new clique is found
            if found_s1
                R = [];
            else
                R = S1;
            end
        else
            % Check the remaining possible combinations of the neighbors
            % indices
            if isempty(find(S2>=max(S1),1))
                R = [];
            else
                R = [];
                for w = find(S2>=max(S1),1):length(S2)
                    S2_aux = S2;
                    S1_aux = S1;
                    S1_aux = [S1_aux S2_aux(w)];
                    S2_aux = setdiff(S2_aux(C(S2(w),S2_aux)==1),S2_aux(w));
                    R = [R;transfer_nodes(S1_aux,S2_aux,clique_size,C)];
                end
            end
        end
    end
end
        

