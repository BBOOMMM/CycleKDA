### CycleKDA

feature和label, 这个数据是一个长度是237的数据扩到711的，所以在你的哪个setting里就是3步step一次，你可以先取x[::3]训练一个baseline, 然后repeat 3次，再和你的方法做对比。评价指标是ic,rank ic和ir, 里面有个index.h5是date x code的格式， 评价的时候就是你产生了某一天全部股票的预测，比如是5000x711的矩阵， 然后你和label每一个时间步算一个相关系数，算了711遍，然后取平均得到今天的评价指标。

baseline: features[:, ::3, :], labels[:, ::3, :]