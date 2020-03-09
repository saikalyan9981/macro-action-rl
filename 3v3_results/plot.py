import matplotlib.pyplot as plt

M = {'action_space_sarsa': {25000: [0.23850000000000005], 40000: [0.254], 45000: [0.24150000000000005], 30000: [0.241], 20000: [0.23050000000000004], 35000: [0.238], 10000: [0.22299999999999998], 50000: [0.23350000000000004], 15000: [0.22550000000000003], 5000: [0.22450000000000003]}, 'sarsa': {35000: [0.14600000000000002], 20000: [0.16700000000000004], 15000: [0.16441779110444776], 40000: [0.15600000000000003], 5000: [0.14549999999999996], 50000: [0.18000000000000005], 10000: [0.16149999999999998], 30000: [0.14935064935064934], 25000: [0.16100000000000003], 45000: [0.14800000000000002]}, 'di_sarsa': {45000: [0.3105], 35000: [0.30300000000000005], 15000: [0.24], 25000: [0.29400000000000004], 50000: [0.3085], 30000: [0.28600000000000003], 10000: [0.20950000000000002], 40000: [0.32599999999999996], 20000: [0.2895], 5000: [0.24450000000000005]}}
M={'3v3_di_sarsa_sa':{5000:[0.52], 10000:[0.501], 15000:[0.489], 20000:[0.515], 25000:[0.51],30000: [0.5185],35000:[0.4965],40000:[0.504],45000: [0.514],50000: [0.494]},'3v3_di_sarsa_ta':{5000:[0.375], 10000:[0.1985], 15000:[0.31100], 20000:[0.352], 25000:[0.312],30000: [0.35750000000000004],35000:[0.31000000000000005],40000:[0.33999999999999997],45000: [0.3035],50000: [0.3005]}}

# M={'normal_di_sarsa_sa':{5000:[0.52], 10000:[0.501], 15000:[0.489], 20000:[0.515], 25000:[0.51],30000: [0.5185],35000:[0.4965],40000:[0.504],45000: [0.514],50000: [0.494]},'regularized_reward_sarsa_sa':{5000:[0.55], 10000:[0.529], 15000:[0.5409999999999999], 20000:[0.528], 25000:[0.536],30000: [0.5549999999999999],35000:[0.542999999999999],40000:[0.5449999999999999],45000: [0.5495],50000: [0.53499999]}}

M['helios'] = {}

M['random_sa'] = {}

for i in range(5000, 55000, 5000):
    M['helios'][i] =  [0.6]
    M['random_sa'][i]=[0.37]


print(M)

for algo in M:
	D = M[algo]
	X = []
	Y = []
	for x in sorted(D):
		X.append(x)
		Y.extend(D[x])
	print(X, Y)
	plt.plot(X, Y, marker='o',label=algo)

plt.ylim((0, 0.7))
plt.legend()
plt.grid(True)

plt.show()