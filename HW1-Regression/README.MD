# Creating J0 with noise
```
lenOfx = 50000
legend = []
X = []
for k in range(lenOfx):
    X.append(8/lenOfx * k)
X = np.array(X)
J0 = []
for x in X:
    if x == 0:
        J0.append(1)
    else:
        J0.append(np.sin(x)/x)
J0 = np.array(J0)

Y = []
for x in X:
    if x == 0:
        Y.append(1 + np.random.normal(0,1) * 0.1)
    else:
        Y.append((np.sin(x)/x) + np.random.normal(0,1) * 0.1)
Y = np.array(Y)

plt.plot(X,J0)
legend.append("J0")
plt.plot(X,Y,linewidth = 0.1,alpha=0.2)
legend.append("J0 (Noisy)")
```
