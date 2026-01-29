

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Load & chuẩn bị dữ liệu
# ------------------------------------------------------------

df_ls_train_original    = pd.read_excel("Limited_LS_train.xlsx")      # Landslide (train gốc)
df_ls_train_generate    = pd.read_excel("CopulaGAN_340.xlsx")      # Landslide (synthetic)
df_ls_test_original     = pd.read_excel("Test_LS.xlsx")                    # Landslide (test)

df_nonls_train_original = pd.read_excel("Limited_NL_train.xlsx")     # Non-landslide (train)
df_nonls_test_original  = pd.read_excel("Test_NL.xlsx")                    # Non-landslide (test)


# Gán nhãn
df_ls_train_generate["Label"]    = 1
df_ls_train_original["Label"]    = 1
df_ls_test_original["Label"]     = 1

df_nonls_train_original["Label"] = 0
df_nonls_test_original["Label"]  = 0

label_col = "Label"

# ------------------------------------------------------------
# 2. Ghép lại tập TRAIN & TEST cho mô hình
# ------------------------------------------------------------

# TRAIN: LS (original + CTGAN) + Non-LS train
train_df = pd.concat(
    [df_ls_train_original, df_ls_train_generate, df_nonls_train_original],
    ignore_index=True
)

# TEST: LS test + Non-LS test
test_df = pd.concat(
    [df_ls_test_original, df_nonls_test_original],
    ignore_index=True
)

print("\n==> Final dataset:")
print("TRAIN total  :", train_df.shape)
print("TEST total   :", test_df.shape)

# ------------------------------------------------------------
# 3. Tách X, y cho TRAIN & TEST
# ------------------------------------------------------------

X_train = train_df.drop(columns=[label_col]).values
y_train = train_df[label_col].astype(int).values

X_test  = test_df.drop(columns=[label_col]).values
y_test  = test_df[label_col].astype(int).values

feature_names = train_df.drop(columns=[label_col]).columns.tolist()
print("\nFeature count:", len(feature_names))

# ------------------------------------------------------------
# 4. XỬ LÝ IMBALANCE: scale_pos_weight cho XGBoost
# ------------------------------------------------------------

counter = Counter(y_train)
n_neg = counter.get(0, 0)
n_pos = counter.get(1, 0)
print("\nClass distribution in TRAIN:", counter)

if n_pos > 0:
    SCALE_POS_WEIGHT = n_neg / n_pos
else:
    SCALE_POS_WEIGHT = 1.0

print(f"scale_pos_weight (for XGBoost) = {SCALE_POS_WEIGHT:.3f}")

# ============================================================
# 5. Định nghĩa không gian hyperparameter (vector con sói)
# ============================================================

# [n_estimators, max_depth, learning_rate, subsample,
#  colsample_bytree, min_child_weight, gamma, reg_lambda]

lb = np.array([100, 10, 0.001, 0.1, 0.5, 1,   0.0,  0.0], dtype=float)
ub = np.array([800, 50, 0.5,   0.5, 1.0, 10,  5.0, 10.0], dtype=float)

dim = len(lb)

def decode_wolf(position):
    """Giải mã 1 con sói GWO thành bộ hyperparameters XGBoost."""
    pos = np.clip(position, lb, ub)

    params = {
        "n_estimators":     int(round(pos[0])),
        "max_depth":        int(round(pos[1])),
        "learning_rate":    float(pos[2]),
        "subsample":        float(pos[3]),
        "colsample_bytree": float(pos[4]),
        "min_child_weight": float(pos[5]),
        "gamma":            float(pos[6]),
        "reg_lambda":       float(pos[7]),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
        "tree_method": "hist",
        "random_state": 10,
        # rất quan trọng cho dữ liệu mất cân bằng
        "scale_pos_weight": SCALE_POS_WEIGHT
    }
    return params

# ============================================================
# 6. Hàm fitness: tối ưu CV F1-score (thay vì Accuracy)
# ============================================================

def fitness(position, X, y, cv_splits=10):
    """
    Đánh giá 1 con sói:
    - Dùng cross-validation F1-score (phù hợp imbalance)
    - fitness = -F1 (để minimize)
    """
    params = decode_wolf(position)
    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=10)

    f1_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )
    mean_f1 = f1_scores.mean()
    return -mean_f1, mean_f1   # fitness, F1

# ============================================================
# 7. Grey Wolf Optimizer (GWO)
# ============================================================

def gwo_optimize(n_wolves, max_iter, X, y):
    """
    Grey Wolf Optimizer để tối ưu hyperparameters XGBoost.
    Trả về:
      - alpha_pos : vector tham số tốt nhất
      - alpha_fit : fitness nhỏ nhất
      - alpha_f1  : CV F1-score tương ứng
    """
    # Khởi tạo quần thể sói ngẫu nhiên
    wolves = np.random.uniform(lb, ub, size=(n_wolves, dim))
    fitness_values = np.zeros(n_wolves)
    f1_values = np.zeros(n_wolves)

    # Đánh giá ban đầu
    for i in range(n_wolves):
        f, f1 = fitness(wolves[i], X, y)
        fitness_values[i] = f
        f1_values[i] = f1

    # Sắp xếp theo fitness
    indices = np.argsort(fitness_values)

    alpha_pos   = wolves[indices[0]].copy()
    alpha_fit   = fitness_values[indices[0]]
    alpha_f1    = f1_values[indices[0]]

    beta_pos    = wolves[indices[1]].copy()
    beta_fit    = fitness_values[indices[1]]

    delta_pos   = wolves[indices[2]].copy()
    delta_fit   = fitness_values[indices[2]]

    print(f"\nInitial alpha CV F1-score (GWO): {alpha_f1:.4f}")

    # Vòng lặp tối ưu
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)   # a: 2 -> 0

        for i in range(n_wolves):
            X_pos = wolves[i].copy()

            for d in range(dim):
                # So với alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - X_pos[d])
                X1 = alpha_pos[d] - A1 * D_alpha

                # So với beta
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[d] - X_pos[d])
                X2 = beta_pos[d] - A2 * D_beta

                # So với delta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[d] - X_pos[d])
                X3 = delta_pos[d] - A3 * D_delta

                X_pos[d] = (X1 + X2 + X3) / 3.0

            # Bound
            X_pos = np.clip(X_pos, lb, ub)
            wolves[i] = X_pos

        # Đánh giá lại
        for i in range(n_wolves):
            f, f1 = fitness(wolves[i], X, y)
            fitness_values[i] = f
            f1_values[i] = f1

        indices = np.argsort(fitness_values)

        # Alpha
        if fitness_values[indices[0]] < alpha_fit:
            alpha_fit = fitness_values[indices[0]]
            alpha_pos = wolves[indices[0]].copy()
            alpha_f1  = f1_values[indices[0]]

        # Beta
        if fitness_values[indices[1]] < beta_fit:
            beta_fit = fitness_values[indices[1]]
            beta_pos = wolves[indices[1]].copy()

        # Delta
        if fitness_values[indices[2]] < delta_fit:
            delta_fit = fitness_values[indices[2]]
            delta_pos = wolves[indices[2]].copy()

        print(f"Iter {t+1:02d}/{max_iter} | Best CV F1-score (GWO): {alpha_f1:.4f}")

    return alpha_pos, alpha_fit, alpha_f1

# ============================================================
# 8. Chạy GWO
# ============================================================

n_wolves = 15
max_iter = 20

best_pos, best_fit, best_f1_cv = gwo_optimize(
    n_wolves=n_wolves,
    max_iter=max_iter,
    X=X_train,
    y=y_train
)

print("\n====================================")
print("GWO Finished")
print(f"Best CV F1-score (train): {best_f1_cv:.4f}")
best_params = decode_wolf(best_pos)
print("Best Hyperparameters (GWO):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ============================================================
# 9. Train mô hình cuối & đánh giá trên TEST
# ============================================================

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.6).astype(int)

# Metrics TEST
test_auc       = roc_auc_score(y_test, y_proba)
acc            = accuracy_score(y_test, y_pred)
f1             = f1_score(y_test, y_pred)
precision      = precision_score(y_test, y_pred)
recall         = recall_score(y_test, y_pred)
cm             = confusion_matrix(y_test, y_pred)
cls_report_duy = classification_report(y_test, y_pred, digits=4)

avg_precision  = average_precision_score(y_test, y_proba)

print("\n====================================")
print("Test Set Performance (threshold = 0.5)")
print(f"Accuracy          : {acc:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"AUC (ROC)         : {test_auc:.4f}")
print(f"Average Precision : {avg_precision:.4f}  (area under PR curve)")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cls_report_duy)

# ROC curve - TEST
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost+GWO (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Landslide Susceptibility Model (GWO-XGBoost, Test)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_landslide_xgb_gwo_test.png", dpi=300)

# ============================================================
# 10. Đánh giá trên TRAIN (xem overfitting)
# ============================================================

# y_proba_train = best_model.predict_proba(X_train)[:, 1]
# y_pred_train  = (y_proba_train >= 0.4).astype(int)

# Train_auc        = roc_auc_score(y_train, y_proba_train)
# acc_train        = accuracy_score(y_train, y_pred_train)
# f1_train         = f1_score(y_train, y_pred_train)
# precision_train  = precision_score(y_train, y_pred_train)
# recall_train     = recall_score(y_train, y_pred_train)
# cm_train         = confusion_matrix(y_train, y_pred_train)
# cls_report_train = classification_report(y_train, y_pred_train, digits=4)

# avg_precision_tr = average_precision_score(y_train, y_proba_train)

# print("\n====================================")
# print("Train Set Performance (threshold = 0.5)")
# print(f"Accuracy Train          : {acc_train:.4f}")
# print(f"F1-score Train          : {f1_train:.4f}")
# print(f"Precision Train         : {precision_train:.4f}")
# print(f"Recall Train            : {recall_train:.4f}")
# print(f"AUC (ROC) Train         : {Train_auc:.4f}")
# print(f"Average Precision Train : {avg_precision_tr:.4f}")
# print("\nConfusion Matrix Train:")
# print(cm_train)
# print("\nClassification Report Train:")
# print(cls_report_train)

# # ROC curve - TRAIN
# fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_proba_train)
# plt.figure()
# plt.plot(fpr_train, tpr_train, label=f"XGBoost+GWO (AUC = {Train_auc:.3f})")
# plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Landslide Susceptibility Model (GWO-XGBoost, Train)")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("roc_curve_landslide_xgb_gwo_train.png", dpi=300)
