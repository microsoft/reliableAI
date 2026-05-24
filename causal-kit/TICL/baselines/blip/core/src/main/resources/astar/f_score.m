function score = f_score(total_path, M)
% total_path is a matrix representing the order we build the k-tree
% M is the mutual information matrix

n = size(M,1);
n_clique = size(total_path, 1);

% compute the g_score
g_score = - sum(sum(M(total_path(1,:),total_path(1,:)))) / 2;
for thread = 2:n_clique
    g_score = g_score - sum(M(total_path(thread,1), total_path(thread,2:end)));
end

% compute the h_score
remain = setdiff(1:n,total_path(:, 1));
% h_score = - sum(max(M(:, remain))); % could be the maximum k, here is 1
h_score = - sum(M(:, remain)); % could be the maximum k, here is 1

score = g_score + h_score;






