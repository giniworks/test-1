import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def generate_not_purchased_data(df):
    # 실제 빈도수 계산하여 customer_frequency, product_frequency 컬럼 추가
    df['customer_frequency'] = df.groupby('AUTH_CUSTOMER_ID')['AUTH_CUSTOMER_ID'].transform('count')
    df['product_frequency'] = df.groupby('PRODUCT_CODE')['PRODUCT_CODE'].transform('count')
    
    # 모든 고유한 고객과 제품 정보 추출
    unique_customers = df['AUTH_CUSTOMER_ID'].unique()
    unique_products = df['PRODUCT_CODE'].unique()
    
    # 빈 리스트 생성
    not_purchased_products_list = []
    
    # 각 고객별로 구매하지 않은 제품 정보 추가
    for customer in unique_customers:
        customer_frequency = df[df['AUTH_CUSTOMER_ID'] == customer]['customer_frequency'].iloc[0]
        purchased_products = df[df['AUTH_CUSTOMER_ID'] == customer]['PRODUCT_CODE'].unique()

        customer_birth_category = df[df['AUTH_CUSTOMER_ID'] == customer]['Birth_Category'].iloc[0]
        customer_gender_category = df[df['AUTH_CUSTOMER_ID'] == customer]['gender_category'].iloc[0]
        
        not_purchased_products = [product for product in unique_products if product not in purchased_products]
        
        not_purchased_products_data = [{'AUTH_CUSTOMER_ID': customer, 
                                        'PRODUCT_CODE': product,
                                        'Birth_Category': customer_birth_category,
                                        'gender_category': customer_gender_category,
                                        'customer_frequency': customer_frequency,
                                        'product_frequency': df[df['PRODUCT_CODE'] == product]['product_frequency'].iloc[0]} 
                                       for product in not_purchased_products]
        
        not_purchased_products_list.extend(not_purchased_products_data)
    
    # 새로운 데이터프레임 생성
    not_purchased_df = pd.DataFrame(not_purchased_products_list)
    not_purchased_df['target'] = 0
    
    return not_purchased_df


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, num_factors, lr=0.01, weight_decay=0.01):
        super(FactorizationMachine, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.w = nn.Parameter(torch.randn(num_features))
        self.v = nn.Parameter(torch.randn(num_features, num_factors))
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        #self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func =  nn.MSELoss()
        self.weight_decay = weight_decay

    def forward(self, x):
        linear_terms = torch.matmul(x, self.w)
        interactions = 0.5 * torch.sum(
            torch.matmul(x, self.v) ** 2 - torch.matmul(x ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )
        return linear_terms + interactions.squeeze()

    def loss(self, y_pred, y_true, c_values):
        mse = (y_pred - y_true.float()) ** 2
        weighted_mse = c_values * mse
        l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2  # L2 regularization
        return torch.mean(weighted_mse) + self.weight_decay * l2_reg

    def train_step(self, x, y, c_values):
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y, c_values)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def recommend_top_n_items(self, user_features, all_item_features, all_item_ids, top_n=5):
        combined_features = torch.cat([user_features.expand(all_item_features.shape[0], -1), all_item_features], dim=1)
        with torch.no_grad():
            scores = self.forward(combined_features)
        sorted_indices = torch.argsort(scores, descending=True)[:top_n]
        return [all_item_ids[i] for i in sorted_indices]

    def recommend_top_n_items_for_all_users(self, user_features_list, all_item_features, all_item_ids, top_n=5):
        recommendations = {}
        for i, user_features in enumerate(user_features_list):
            user_id = i  # or replace with actual user ID if you have that info
            top_n_items = self.recommend_top_n_items(user_features, all_item_features, all_item_ids, top_n)
            recommendations[user_id] = top_n_items
        return recommendations

# Preprocessing
target_col = 'target'
#df, label_encoders = process_dataframe(df, target_col)
X = df.drop(columns=[target_col,'PRODUCT_CODE','C','AUTH_CUSTOMER_ID'])
y = df[target_col]
c = df['C']
num_epochs = 10
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1)

c_values_tensor = torch.tensor(c, dtype=torch.float32)
c_values_tensor = torch.where(c_values_tensor < 1, c_values_tensor * 100, c_values_tensor)


# User features
unique_user_df = df.drop_duplicates(subset=['AUTH_CUSTOMER_ID']).sort_values('AUTH_CUSTOMER_ID')
user_features_df = unique_user_df[['Birth_Category', 'gender_category']]
user_feature_tensor = torch.tensor(pd.get_dummies(user_features_df).values, dtype=torch.float32)

# Item features
unique_item_df = df.drop_duplicates(subset=['PRODUCT_CODE']).sort_values('PRODUCT_CODE')
item_features_df = unique_item_df.filter(like='DEPTH')
item_feature_tensor = torch.tensor(item_features_df.values, dtype=torch.float32)


# item_ids
all_item_ids = list(df.PRODUCT_CODE.unique())

# Initialize model
num_features = X.shape[1] 
#num_features = user_feature_tensor.shape[1] + item_feature_tensor.shape[1]
num_factors = 3
model = FactorizationMachine(num_features, num_factors)


# # Dummy Training loop


for epoch in range(num_epochs):
    loss = model.train_step(X_tensor, y_tensor,c_values_tensor)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

# Make recommendations
recommendations = model.recommend_top_n_items_for_all_users(user_feature_tensor, item_feature_tensor, all_item_ids, top_n=5)
print("User-wise top 5 recommended items:", recommendations)
