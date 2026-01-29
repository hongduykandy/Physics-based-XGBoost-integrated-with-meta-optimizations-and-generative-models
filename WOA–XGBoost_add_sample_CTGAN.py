# ============================================================
# XGBoost + Whale Optimization Algorithm (WOA) for Landslide Assessment
# - Input:
#   + df_ls_train_generate    = LS_2_aftercheck_CTGAN_340.xlsx  (LS synthetic)
#   + df_ls_train_original    = 68_limited_point_LS_train.xlsx  (LS train gốc)
#   + df_ls_test_original     = Test_LS_30%.xlsx                (LS test)
#   + df_nonls_train_original = 70%_limited_point_NL_train.xlsx (Non-LS train)
#   + df_nonls_test_original  = Test_NL_30%.xlsx                (Non-LS test)
# - Train WOA + XGBoost trên train_df (có imbalance)
# - Xử lý IMBALANCE:
#     * scale_pos_weight = n_negative / n_positive
#     * fitness = F1-score (thay vì Accuracy)
# - Đánh giá: AUC, Accuracy, F1, Precision, Recall, ROC, PR curve
# ============================================================

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
# 5. Ghép lại tập TRAIN & TEST cho mô hình
# ------------------------------------------------------------

# TRAIN: LS (CTGAN + original) + Non-LS train
train_df = pd.concat(
    [ df_ls_train_original,df_ls_train_generate, df_nonls_train_original],
    ignore_index=True
)

# TEST: LS test + Non-LS test
test_df = pd.concat(
    [df_ls_test_original, df_nonls_test_original],
    ignore_index=True
)

print("\n==> Tập cuối cùng:")
print("TRAIN total  :", train_df.shape)
print("TEST total   :", test_df.shape)

# ------------------------------------------------------------
# 6. Tách X, y cho TRAIN & TEST
# ------------------------------------------------------------

X_train = train_df.drop(columns=[label_col]).values
y_train = train_df[label_col].astype(int).values

X_test  = test_df.drop(columns=[label_col]).values
y_test  = test_df[label_col].astype(int).values

feature_names = train_df.drop(columns=[label_col]).columns.tolist()
print("\nFeature count:", len(feature_names))

# ------------------------------------------------------------
# XỬ LÝ IMBALANCE: scale_pos_weight cho XGBoost
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
# 7. Định nghĩa không gian hyperparameter
# ============================================================

# [n_estimators, max_depth, learning_rate, subsample,
#  colsample_bytree, min_child_weight, gamma, reg_lambda]

lb = np.array([100, 10, 0.001, 0.1, 0.5, 1,   0.0,  0.0], dtype=float)
ub = np.array([800, 50, 0.5, 0.5, 1.0, 10,  5.0, 10.0], dtype=float)

dim = len(lb)

def decode_whale(position):
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
        # 🔴 Quan trọng cho dữ liệu mất cân bằng:
        "scale_pos_weight": SCALE_POS_WEIGHT
    }
    return params

# ============================================================
# 8. Hàm fitness: tối ưu CV F1-score (thay vì Accuracy)
# ============================================================

def fitness(position, X, y, cv_splits=10):
    """
    Đánh giá 1 cá voi:
    - Dùng cross-validation F1-score (phù hợp imbalance)
    - fitness = -F1 (để minimize)
    """
    params = decode_whale(position)
    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=10)

    f1_scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="f1",    # 🔴 Đổi từ "accuracy" sang "f1"
        n_jobs=-1
    )
    mean_f1 = f1_scores.mean()
    return -mean_f1, mean_f1   # fitness, F1

# ============================================================
# 9. Whale Optimization Algorithm (WOA)
# ============================================================

def woa_optimize(n_whales, max_iter, X, y):
    """
    Whale Optimization Algorithm để tối ưu hyperparameters XGBoost.
    Trả về:
      - best_pos   : vector tham số tốt nhất
      - best_fit   : fitness tốt nhất (nhỏ nhất)
      - best_acc   : CV F1-score tương ứng
    """
    # Khởi tạo quần thể cá voi
    whales = np.random.uniform(lb, ub, size=(n_whales, dim))
    fitness_values = np.zeros(n_whales)
    f1_values = np.zeros(n_whales)

    # Đánh giá ban đầu
    for i in range(n_whales):
        f, f1 = fitness(whales[i], X, y)
        fitness_values[i] = f
        f1_values[i] = f1

    # Xác định cá voi tốt nhất
    best_idx = np.argmin(fitness_values)
    best_pos = whales[best_idx].copy()
    best_fit = fitness_values[best_idx]
    best_f1 = f1_values[best_idx]

    print(f"\nInitial best CV F1-score: {best_f1:.4f}")

    b = 0.9  # tham số cho chuyển động xoắn ốc

    # Vòng lặp chính
    for t in range(max_iter):
        # a giảm tuyến tính từ 2 -> 0
        a = 2 - 2 * t / max_iter

        for i in range(n_whales):
            X_pos = whales[i].copy()

            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()

            if p < 0.5:
                if abs(A) < 1:
                    # Khai thác: bao vây con mồi tốt nhất
                    D = np.abs(C * best_pos - X_pos)
                    X_new = best_pos - A * D
                else:
                    # Khai phá: chọn 1 cá voi ngẫu nhiên
                    rand_idx = np.random.randint(0, n_whales)
                    X_rand = whales[rand_idx]
                    D = np.abs(C * X_rand - X_pos)
                    X_new = X_rand - A * D
            else:
                # Chuyển động xoắn ốc hướng tới con mồi tốt nhất
                D_prime = np.abs(best_pos - X_pos)
                l = np.random.uniform(-1, 1)
                X_new = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            # Áp dụng bound
            X_new = np.clip(X_new, lb, ub)
            whales[i] = X_new

        # Đánh giá lại sau khi cập nhật vị trí
        for i in range(n_whales):
            f, f1 = fitness(whales[i], X, y)
            fitness_values[i] = f
            f1_values[i] = f1

        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fit:
            best_fit = fitness_values[current_best_idx]
            best_pos = whales[current_best_idx].copy()
            best_f1 = f1_values[current_best_idx]

        print(f"Iter {t+1:02d}/{max_iter} | Best CV F1-score: {best_f1:.4f}")

    return best_pos, best_fit, best_f1

# ============================================================
# 10. Chạy WOA
# ============================================================

n_whales = 15
max_iter =30
best_pos, best_fit, best_f1_cv = woa_optimize(
    n_whales=n_whales,
    max_iter=max_iter,
    X=X_train,
    y=y_train
)

print("\n====================================")
print("WOA Finished")
print(f"Best CV F1-score (train): {best_f1_cv:.4f}")
best_params = decode_whale(best_pos)
print("Best Hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ============================================================
# 11. Train mô hình cuối & đánh giá trên TEST
# ============================================================

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.6).astype(int)

# Metrics
test_auc       = roc_auc_score(y_test, y_proba)
acc            = accuracy_score(y_test, y_pred)
f1             = f1_score(y_test, y_pred)
precision      = precision_score(y_test, y_pred)
recall         = recall_score(y_test, y_pred)
cm             = confusion_matrix(y_test, y_pred)
cls_report_duy = classification_report(y_test, y_pred, digits=4)

# prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

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

# ----------------- ROC curve -----------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"XGBoost+WOA (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Landslide Susceptibility Model (WOA-XGBoost)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_landslide_xgb_woa.png", dpi=300)




# 12. Train mô hình cuối & đánh giá trên TRAIN
# ============================================================

# y_proba_train = best_model.predict_proba(X_train)[:, 1]
# y_pred_train  = (y_proba_train >= 0.4).astype(int)

# # Metrics
# Train_auc       = roc_auc_score(y_train, y_proba_train)
# acc_train           = accuracy_score(y_train, y_pred_train)
# f1_train            = f1_score(y_train, y_pred_train)
# precision_train      = precision_score(y_train, y_pred_train)
# recall_train         = recall_score(y_train, y_pred_train)
# cm_train             = confusion_matrix(y_train, y_pred_train)
# cls_report_train = classification_report(y_train, y_pred_train, digits=4)

# prec_curve_train, rec_curve_train, pr_thresholds_train = precision_recall_curve(y_train, y_proba_train)
# avg_precision = average_precision_score(y_train, y_proba_train)

# print("\n====================================")
# print("Train Set Performance (threshold = 0.5)")
# print(f"AccuracyTrain          : {acc:.4f}")
# print(f"F1-score Train           : {f1:.4f}")
# print(f"Precision  Train       : {precision:.4f}")
# print(f"Recall    Train        : {recall:.4f}")
# print(f"AUC (ROC) Train        : {Train_auc:.4f}")
# print(f"Average Precision Train : {avg_precision:.4f}  (area under PR curve)")
# print("\nConfusion Matrix Train:")
# print(cm)
# print("\nClassification Report Train:")
# print(cls_report_train)

# # ----------------- ROC curve -----------------
# fpr_train, tpr_train, thresholds = roc_curve(y_train, y_proba_train)

# plt.figure()
# plt.plot(fpr_train, tpr_train, label=f"XGBoost+WOA (AUC = {Train_auc:.3f})")
# plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Landslide Susceptibility Model (WOA-XGBoost) train")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("roc_curve_landslide_xgb_woa.png", dpi=300)


