import numpy as np
import matplotlib.pyplot as plt

class RealEstateRegressor:
    """
    نموذج احترافي للتنبؤ بأسعار العقارات مبني من الصفر.
    تم تصميمه ليكون تعليمياً وعملياً في نفس الوقت.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.epochs = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        # التحقق من أبعاد البيانات وتحويلها لمصفوفات NumPy
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        # 1. تهيئة الأوزان (Initialization)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        print(f"--- بدء عملية التدريب الذكي ---")
        
        for i in range(self.epochs):
            # 2. التوقع (Prediction Model)
            y_approximated = np.dot(X, self.weights) + self.bias

            # 3. حساب الخطأ (Mean Squared Error)
            cost = (1 / (2 * n_samples)) * np.sum((y_approximated - y)**2)
            self.loss_history.append(cost)

            # 4. حساب الاشتقاقات (Gradients Calculation)
            dw = (1 / n_samples) * np.dot(X.T, (y_approximated - y))
            db = (1 / n_samples) * np.sum(y_approximated - y)

            # 5. تحديث القيم (Weights Update)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"دورة {i}: مقدار الخطأ الحالي = {cost:.4f}")

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias

    def visualize_results(self, X, y):
        """دالة جمالية لعرض النتائج بيانيًا"""
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='البيانات الحقيقية')
        plt.plot(X, self.predict(X), color='red', linewidth=2, label='خط التوقع (Model)')
        plt.title('تحليل أسعار العقارات: المساحة مقابل السعر')
        plt.xlabel('المساحة (بالوحدة)')
        plt.ylabel('السعر (بالوحدة)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- تشغيل النموذج على بيانات تجريبية ---

# بيانات (المساحة بـ 100 متر مربع ، السعر بالمليون)
X_train = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y_train = [1.1, 1.6, 2.3, 2.8, 3.2, 3.9, 4.5]

# إنشاء وتشغيل النموذج
model = RealEstateRegressor(learning_rate=0.05, iterations=1000)
model.fit(X_train, y_train)

# توقع سعر عقار جديد مساحته 500 متر (القيمة 5.0)
test_area = 5.0
predicted_price = model.predict(test_area)

print("\n" + "="*30)
print(f"النتيجة النهائية:")
print(f"السعر المتوقع لمساحة {test_area} هو: {predicted_price[0][0]:.2f} مليون")
print("="*30)

# عرض الرسم البياني (اختياري - ابهر به توفيق)
model.visualize_results(X_train, y_train) 