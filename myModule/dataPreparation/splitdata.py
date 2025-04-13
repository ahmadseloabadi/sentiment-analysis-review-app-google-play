
from sklearn.model_selection import train_test_split
def data_spilt(kolom_ulasan,kolom_label,test_size):
    x=kolom_ulasan
    y=kolom_label
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
    return X_train, X_test, Y_train, Y_test