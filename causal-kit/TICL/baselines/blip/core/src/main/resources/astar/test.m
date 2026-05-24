n = 20

M = rand(n)

M = (M + M')/2

M(1:n+1:n*n) = 0;

R = [1,2,3]

AstarKtree(R, M)
