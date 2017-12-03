# Combination

# Combination(n, k), eg.Combination(7, 2)

def Combination(n, k):
	if k < 0 or k > n:
		print('Please input the right number!')
		print('0 < k < n ')
	else:
		p, q = 1, 1
		for i in range(1, min(k, n - k) + 1):
			p *= n
			q *= i
			n -= 1
		return p // q

print(Combination(7, 2))