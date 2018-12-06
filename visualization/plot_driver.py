import matplotlib.pyplot as plt
driver_number_accuracy_dict = {9: 0.9300000000000003, 8: 0.9166666666666669, 7: 0.875, 5: 0.8383333333333334,
                               4: 0.8228571428571424, 3: 0.76625, 2: 0.735555555555555, 1: 0.6900000000000001, 0: 0.6763636363636357}
X = list(driver_number_accuracy_dict.keys())[::-1]
y = [driver_number_accuracy_dict[k] for k in X]
plt.plot(X, y, 'k-o')
plt.xticks(range(10))
plt.xlabel('Number of Driver being excluded')
plt.ylabel('Mean Error in 100 experiment')
plt.savefig('../figures/Learn2classify_test/RandomForest_drive_change_error.png')
