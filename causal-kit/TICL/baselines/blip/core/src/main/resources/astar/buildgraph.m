function G = buildgraph(P,n)
% build a graph according to the path

G = eye(n);
l = size(P, 1);
G(P(1, :), P(1, :)) = 1;
if l > 1
    for thread = 2:l
        G(P(thread,1), P(thread, 2:end)) = 1;
        G(P(thread, 2:end), P(thread,1)) = 1;
    end
end