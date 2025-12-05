import torch
import math
# Bài 1 
# Ma trận đầu vào A (hàng = Thời tiết, cột = Địa điểm)
A = torch.tensor([
    [1, 2, 3, 4],   # Nắng
    [3, 4, 5, 6],   # Mưa
    [2, 3, 5, 6],   # Âm u
    [1, 3, 5, 7]    # Lạnh
], dtype=torch.float32)

# ---------------------------------------------------------
# 1) Tính phân phối xác suất chung P(Weather, Location)
# ---------------------------------------------------------

# Tổng tất cả phần tử trong ma trận (để chuẩn hóa về xác suất)
total_sum = A.sum()

# Chia từng phần tử cho tổng → ma trận xác suất chung
P_joint = A / total_sum


# ---------------------------------------------------------
# 2) Tính phân phối biên theo THỜI TIẾT: P(Weather)
# ---------------------------------------------------------
# Cộng theo từng hàng (dim=1 = tổng theo cột)
P_weather = P_joint.sum(dim=1)


# ---------------------------------------------------------
# 3) Tính phân phối biên theo ĐỊA ĐIỂM: P(Location)
# ---------------------------------------------------------
# Cộng theo từng cột (dim=0 = tổng theo hàng)
P_location = P_joint.sum(dim=0)


# ---------------------------------------------------------
# 4) Gán nhãn cho rõ ràng (không bắt buộc, nhưng giúp dễ đọc)
weather_labels = ['Nắng', 'Mưa', 'Âm u', 'Lạnh']
location_labels = ['Rất gần', 'Gần', 'Xa', 'Rất xa']


# ---------------------------------------------------------
# 5) In kết quả
# ---------------------------------------------------------

print("=== Phân phối xác suất chung P(Weather, Location) ===")
print(P_joint)

print("\n=== Phân phối biên P(Weather) ===")
for lbl, prob in zip(weather_labels, P_weather):
    print(f"{lbl}: {prob:.4f}")

print("\n=== Phân phối biên P(Location) ===")
for lbl, prob in zip(location_labels, P_location):
    print(f"{lbl}: {prob:.4f}")


#######################3######################################
# Bài 2
def multinomial_p(n, counts, probs):
    """
    n: tổng số phép thử
    counts: list số lần quan sát ở từng loại, vd [x1, x2, x3]
    probs: list xác suất từng loại, vd [p1, p2, p3]
    """
    if sum(counts) != n:
        raise ValueError("Tổng counts phải bằng n")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("Tổng xác suất trong p phải bằng 1")

    # n! / (x1! x2! ... xm!)
    coef = math.factorial(n) 
    for x in counts:
        coef /= math.factorial(x)

    # ∏ p_i^{x_i}
    prob_term = 1.0
    for x, p in zip(counts, probs):
        prob_term *= (p ** x)

    return coef * prob_term

# Ví dụ: n=5, 3 loại, counts=[2,2,1], p=[0.2,0.3,0.5]
print(multinomial_p(5, [2, 2, 1], [0.2, 0.3, 0.5]))
