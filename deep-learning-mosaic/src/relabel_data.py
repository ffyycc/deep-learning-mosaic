
def filter_relabel(x_train, y_train, x_test, y_test):
    def append_idx(list, set):
        res = []
        for i in range(len(list)):
            if list[i][0] in set:
                res.append(i)
        return res
    
    def re_idx(list, dic):
        for i in range(len(list)):
            list[i][0] = dic[list[i][0]]
        return list
    
    chosen_set = set([0,1,3,5])
    # get all labels from selected idx
    idx_train = append_idx(y_train, chosen_set)
    idx_test = append_idx(y_test, chosen_set)
    
    x_train_selected = x_train[idx_train]
    y_train_selected = y_train[idx_train]
    x_test_selected = x_test[idx_test]
    y_test_selected = y_test[idx_test]
    
    # relabel the data in range 0-3
    redir_dic = {3:0, 5:1, 0:2, 1:3}
    
    y_train_selected = re_idx(y_train_selected, redir_dic)
    y_test_selected = re_idx(y_test_selected, redir_dic)
    
    return x_train_selected, y_train_selected, x_test_selected, y_test_selected