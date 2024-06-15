from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2')) # 标准化+L2正则化
    scores = cross_val_score(model, X_train, y_train, cv=6) # 交叉验证仅用于评估模型的性能，并不能直接训练模型参数
    print("交叉验证得分:", scores)
    print("平均交叉验证得分:", scores.mean())

    model.fit(X_train, y_train) # 完成交叉验证后，为了获得在整个训练集上更全面的训练结果，需要使用model.fit(X_train, y_train)来重新训练模型
    test_score = model.score(X_test, y_test)
    print("测试集准确率:", test_score)

    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(model, X_train, y_train, cv=6, return_train_score=True)
    train_scores = cv_results['train_score']
    test_scores = cv_results['test_score']
    for fold, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
        print(f"Fold {fold + 1}: 训练集得分={train_score}, 测试集得分={test_score}")
    print("平均训练集得分:", train_scores.mean())
    print("平均测试集得分:", test_scores.mean())
